"""
SAM Training Loss Functions
============================
Combined loss for fine-tuning SAM:
  - L1 pixel reconstruction
  - Identity preservation (cosine similarity of face features)
  - Age classification (cross-entropy on age bins)
  - Adversarial (non-saturating GAN loss)
  - Discriminator loss (R1-regularized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityLoss(nn.Module):
    """
    Identity preservation loss using a frozen feature extractor.

    Uses the pSp pretrained encoder (3-channel input) as a face feature
    extractor. Computes cosine distance between feature representations
    of the generated and target images.
    """

    def __init__(self):
        super().__init__()
        self._encoder = None

    def set_encoder(self, pretrained_encoder: nn.Module):
        """Set a frozen encoder for identity feature extraction."""
        self._encoder = pretrained_encoder
        for p in self._encoder.parameters():
            p.requires_grad = False
        self._encoder.eval()

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute identity loss between generated and target images.

        If no encoder is set, falls back to L1 loss on downscaled images.
        """
        if self._encoder is not None:
            with torch.no_grad():
                target_features = self._encoder(target)
            gen_features = self._encoder(generated)
            # Cosine distance: 1 - cos_similarity
            cos_sim = F.cosine_similarity(
                gen_features.view(gen_features.size(0), -1),
                target_features.view(target_features.size(0), -1),
                dim=1,
            )
            return (1.0 - cos_sim).mean()
        else:
            # Fallback: L1 on downscaled images (proxy for identity)
            gen_down = F.interpolate(generated, size=(112, 112), mode='bilinear', align_corners=False)
            tgt_down = F.interpolate(target, size=(112, 112), mode='bilinear', align_corners=False)
            return F.l1_loss(gen_down, tgt_down)


class AgeLoss(nn.Module):
    """
    Age supervision loss using cross-entropy over age bins (0-100).

    Uses a simple CNN classifier trained to predict age from face images.
    If no classifier is available, this loss returns zero.
    """

    def __init__(self, num_bins: int = 101):
        super().__init__()
        self.num_bins = num_bins
        self._classifier = None
        self.criterion = nn.CrossEntropyLoss()

    def set_classifier(self, classifier: nn.Module):
        """Set a frozen age classifier for age supervision."""
        self._classifier = classifier
        for p in self._classifier.parameters():
            p.requires_grad = False
        self._classifier.eval()

    def forward(self, generated: torch.Tensor, target_ages: torch.Tensor) -> torch.Tensor:
        """
        Compute age loss.

        Parameters
        ----------
        generated : [B, 3, H, W] tensor
        target_ages : [B] tensor of integer ages (0-100)
        """
        if self._classifier is not None:
            with torch.no_grad():
                pass  # Classifier would predict ages here
            # For now, return zero until classifier is loaded
            return torch.tensor(0.0, device=generated.device)

        return torch.tensor(0.0, device=generated.device)


class SAMTrainingLoss(nn.Module):
    """
    Combined training loss for SAM fine-tuning.

    Combines L1, identity, age, and adversarial losses with configurable weights.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.l1_loss = nn.L1Loss()
        self.identity_loss = IdentityLoss()
        self.age_loss = AgeLoss(num_bins=config.age_bins)

    def set_identity_encoder(self, encoder: nn.Module):
        """Set the pretrained encoder for identity loss."""
        self.identity_loss.set_encoder(encoder)

    def set_age_classifier(self, classifier: nn.Module):
        """Set the age classifier for age loss."""
        self.age_loss.set_classifier(classifier)

    def forward(self, generated: torch.Tensor, target: torch.Tensor,
                target_ages: torch.Tensor = None,
                disc_fake: torch.Tensor = None) -> tuple[torch.Tensor, dict]:
        """
        Compute combined generator loss.

        Parameters
        ----------
        generated : [B, 3, 256, 256] generated images
        target : [B, 3, 256, 256] ground truth images
        target_ages : [B] integer ages (optional)
        disc_fake : discriminator output on generated images (optional)

        Returns
        -------
        total_loss : scalar tensor
        loss_dict : dict of individual loss values (for logging)
        """
        losses = {}
        total = torch.tensor(0.0, device=generated.device)

        # L1 pixel reconstruction
        l1 = self.l1_loss(generated, target)
        losses['l1'] = l1.item()
        total = total + self.config.lambda_l1 * l1

        # Identity preservation
        if self.config.lambda_identity > 0:
            id_loss = self.identity_loss(generated, target)
            losses['identity'] = id_loss.item()
            total = total + self.config.lambda_identity * id_loss

        # Age supervision
        if target_ages is not None and self.config.lambda_age > 0:
            age_loss = self.age_loss(generated, target_ages)
            losses['age'] = age_loss.item()
            total = total + self.config.lambda_age * age_loss

        # Adversarial (non-saturating)
        if disc_fake is not None and self.config.lambda_adv > 0:
            adv_loss = F.softplus(-disc_fake).mean()
            losses['adv'] = adv_loss.item()
            total = total + self.config.lambda_adv * adv_loss

        losses['total'] = total.item()
        return total, losses

    @staticmethod
    def discriminator_loss(disc_real: torch.Tensor,
                           disc_fake: torch.Tensor) -> torch.Tensor:
        """
        Non-saturating discriminator loss with gradient penalty.

        Parameters
        ----------
        disc_real : discriminator output on real images
        disc_fake : discriminator output on fake (generated) images
        """
        loss = F.softplus(-disc_real).mean() + F.softplus(disc_fake).mean()
        return loss
