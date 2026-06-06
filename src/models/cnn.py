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
