"""Model factory for the IberFire segmentation U-Net.

Single place that constructs the network, so training (``scripts/train.py``) and
serving (``docker/monolith/app.py``) can never drift in architecture. Previously
``smp.Unet(...)`` was instantiated inline in both files.
"""

from typing import Optional

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_unet(
    in_channels: int,
    encoder_name: str = "resnet34",
    classes: int = 1,
    encoder_weights: Optional[str] = "imagenet",
    decoder_dropout: float = 0.0,
    activation: Optional[str] = None,
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
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
        decoder_dropout=decoder_dropout,
    )
