"""Model factory for the IberFire segmentation U-Net.

Single place that constructs the network, so training (``scripts/train.py``) and
serving (``docker/monolith/app.py``) can never drift in architecture. Previously
``smp.Unet(...)`` was instantiated inline in both files.
"""

from typing import Optional

import segmentation_models_pytorch as smp
import torch.nn as nn


def _groupnorm_num_groups(num_channels: int, max_groups: int = 32) -> int:
    """Largest group count from {32,16,8,4,2,1} that divides num_channels."""
    for g in (max_groups, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return g
    return 1


def convert_bn_to_groupnorm(module: nn.Module, max_groups: int = 32) -> nn.Module:
    """Recursively replace every BatchNorm2d with GroupNorm (batch-size-independent).

    Needed when batches are small (here, a few full 4 km images on ~14 GB) — small-batch
    BatchNorm running stats are noisy/unstable; GroupNorm sidesteps it. Pretrained conv
    weights are kept; the (fresh) GroupNorm affine params are learned.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            ng = _groupnorm_num_groups(child.num_features, max_groups)
            setattr(module, name, nn.GroupNorm(ng, child.num_features, affine=True))
        else:
            convert_bn_to_groupnorm(child, max_groups)
    return module


def build_unet(
    in_channels: int,
    encoder_name: str = "resnet34",
    classes: int = 1,
    encoder_weights: Optional[str] = "imagenet",
    decoder_dropout: float = 0.0,
    activation: Optional[str] = None,
    norm: str = "batch",
) -> nn.Module:
    """Construct the segmentation U-Net used across training and inference.

    Args:
        in_channels: Number of input feature channels (e.g. ``len(FEATURE_VARS)``).
        encoder_name: Backbone encoder (e.g. ``"resnet34"``, ``"resnet50"``,
            ``"se_resnext50_32x4d"``).
        classes: Number of output channels (1 for binary fire occurrence).
        encoder_weights: Pretrained weights for the encoder. Use ``"imagenet"``
            when training from scratch; pass ``None`` for inference where a
            checkpoint will overwrite the weights anyway (avoids a download).
        decoder_dropout: Dropout in the decoder. Inert at eval time, so it does
            not affect inference parity or the loadable ``state_dict``.
        activation: Final activation. Keep ``None`` and apply ``sigmoid``
            explicitly so logits flow to the (logit-adjusted) loss.

    Returns:
        An ``smp.Unet`` module. The returned ``state_dict`` is independent of
        ``encoder_weights`` and ``decoder_dropout``, so checkpoints remain
        interchangeable across those settings.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
        decoder_dropout=decoder_dropout,
    )
    if norm == "group":
        convert_bn_to_groupnorm(model)
    elif norm != "batch":
        raise ValueError(f"norm must be 'batch' or 'group', got {norm!r}")
    return model


class WideDeepUNet(nn.Module):
    """Wide-and-deep segmentation: a deep spatial U-Net + a wide point-wise (1x1) branch.

    Motivation (see CHANGES.md, 2026-06-06 baseline): the plain U-Net matches the GBT floor on
    the *spatial* spread regime (~0.99) but underperforms the point-wise GBT on *spatially-sparse*
    new ignitions (~0.32 vs 0.50). Its spatial priors hurt there — the encoder's internal 32x stride
    blurs the per-cell signal and convolutional smoothness kills precision on a spiky target.

    The wide branch is a per-pixel MLP (stacked 1x1 convs: receptive field = ONE cell, NO downsampling),
    i.e. the GBT-style pathway, applied densely so it still emits a full HxW map. Fusion is ADDITIVE on
    logits (z = z_deep + z_wide), and the wide head is ZERO-INITIALIZED so the model starts bit-for-bit
    at the deep U-Net baseline and can only *add* point-wise signal — it cannot regress below it. Each
    branch is independently ablatable (zero one out at eval) to measure its contribution.
    """

    def __init__(self, in_channels: int, wide_hidden=(128, 64), wide_dropout: float = 0.1,
                 classes: int = 1, **unet_kwargs):
        super().__init__()
        self.deep = build_unet(in_channels=in_channels, classes=classes, **unet_kwargs)
        layers = []
        c = in_channels
        for h in wide_hidden:  # per-pixel MLP — every layer is 1x1 (no spatial mixing)
            layers += [nn.Conv2d(c, h, kernel_size=1),
                       nn.GroupNorm(_groupnorm_num_groups(h), h),
                       nn.GELU(),
                       nn.Dropout2d(wide_dropout)]
            c = h
        head = nn.Conv2d(c, classes, kernel_size=1)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)  # start == deep baseline (z_wide = 0 at init)
        layers.append(head)
        self.wide = nn.Sequential(*layers)

    def forward(self, x):
        return self.deep(x) + self.wide(x)


def build_wide_deep_unet(in_channels: int, wide_hidden=(128, 64), wide_dropout: float = 0.1,
                         **kwargs) -> nn.Module:
    """Factory for the wide-and-deep variant (deep U-Net + point-wise 1x1 branch).

    Accepts the same kwargs as ``build_unet`` (encoder_name, encoder_weights, decoder_dropout,
    norm, classes, ...), forwarded to the deep branch. ``wide_hidden``/``wide_dropout`` size and
    regularize the wide per-pixel MLP. The wide head is zero-initialized, so an untrained
    WideDeepUNet is numerically identical to ``build_unet(**kwargs)``.
    """
    return WideDeepUNet(in_channels, wide_hidden=wide_hidden, wide_dropout=wide_dropout, **kwargs)
