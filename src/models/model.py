"""Two-head classification model built on MedSigLIP vision encoder."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel

from dataloader.class_labels import NUM_DR_CLASSES, NUM_EDEMA_CLASSES


@dataclass
class DualHeadOutput:
    """Output container for the dual-head model."""

    loss: torch.Tensor | None = None
    dr_logits: torch.Tensor | None = None
    edema_logits: torch.Tensor | None = None
    dr_loss: torch.Tensor | None = None
    edema_loss: torch.Tensor | None = None


class DualHeadMedSigLIP(nn.Module):
    """MedSigLIP vision encoder with DR and Edema classification heads."""

    def __init__(self, model_id: str, alpha: float = 1.0, beta: float = 1.0, freeze_backbone: bool = False):
        super().__init__()
        base_model = AutoModel.from_pretrained(model_id)
        self.vision_encoder = base_model.vision_model
        hidden_size = self.vision_encoder.config.hidden_size

        self.dr_head = nn.Linear(hidden_size, NUM_DR_CLASSES)
        self.edema_head = nn.Linear(hidden_size, NUM_EDEMA_CLASSES)

        self.criterion = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.beta = beta

        if freeze_backbone:
            self.freeze_backbone()

    def forward(
        self,
        pixel_values: torch.Tensor,
        dr_labels: torch.Tensor | None = None,
        edema_labels: torch.Tensor | None = None,
    ) -> DualHeadOutput:
        """Forward pass through vision encoder and both classification heads."""
        vision_out = self.vision_encoder(pixel_values=pixel_values)
        pooled = vision_out.pooler_output

        dr_logits = self.dr_head(pooled)
        edema_logits = self.edema_head(pooled)

        loss = None
        dr_loss = None
        edema_loss = None

        if dr_labels is not None and edema_labels is not None:
            dr_loss = self.criterion(dr_logits, dr_labels)
            edema_loss = self.criterion(edema_logits, edema_labels)
            loss = self.alpha * dr_loss + self.beta * edema_loss

        return DualHeadOutput(
            loss=loss, dr_logits=dr_logits, edema_logits=edema_logits, dr_loss=dr_loss, edema_loss=edema_loss
        )

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Get the number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def freeze_backbone(self) -> None:
        """Freeze backbone layers, keeping only classification heads trainable."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self._unfreeze_heads()

    def unfreeze_all(self) -> None:
        """Unfreeze all layers for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def _unfreeze_heads(self) -> None:
        """Unfreeze both classification heads."""
        for param in self.dr_head.parameters():
            param.requires_grad = True
        for param in self.edema_head.parameters():
            param.requires_grad = True

    def get_feature_layer(self) -> nn.Module:
        """Get the last encoder layer for feature extraction (Grad-CAM target)."""
        return self.vision_encoder.encoder.layers[-1]
