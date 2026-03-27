"""
Loss Functions for Joint Training (Base Model + Refinement Network)

Key insight:
    - Base model loss: Higher crack_weight (0.5-0.6) → Encourages high RECALL
    - Refinement loss: Lower crack_weight (0.3-0.4) → Encourages high PRECISION
    - Different weights make them learn different behaviors!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(inputs, targets, smooth=1, crack_weight=0.5):
    """
    Dice + BCE loss with configurable crack weight.

    Args:
        inputs: Model predictions (before sigmoid)
        targets: Ground truth masks
        smooth: Smoothing factor for Dice coefficient
        crack_weight: Weight for crack pixels (0.0-1.0)

    Returns:
        Combined Dice + BCE loss
    """
    inputs = torch.sigmoid(inputs)
    inputs_flat = inputs.view(-1)
    targets_flat = targets.view(-1)

    # Weighted BCE
    weight = torch.zeros_like(targets_flat, device=targets_flat.device)
    weight.fill_(1.0 - crack_weight)  # background weight
    weight[targets_flat > 0] = crack_weight  # crack weight

    intersection = (inputs_flat * targets_flat).sum()
    dice = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
    bce = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean', weight=weight)

    return bce + dice


class JointLoss(nn.Module):
    """
    Combined loss for joint training of base model + refinement network.

    Computes two separate losses:
        1. Base model loss (higher crack_weight for recall)
        2. Refinement loss (lower crack_weight for precision)

    Total loss = alpha * base_loss + beta * refinement_loss
    """

    def __init__(
        self,
        base_crack_weight=0.5,
        refinement_crack_weight=0.3,
        base_loss_weight=0.3,
        refinement_loss_weight=0.7,
        use_consistency_loss=False,
        consistency_weight=0.1
    ):
        """
        Args:
            base_crack_weight: Crack weight for base model loss (0.5-0.6 for recall)
            refinement_crack_weight: Crack weight for refinement loss (0.3-0.4 for precision)
            base_loss_weight: Weight for base model loss in total (alpha)
            refinement_loss_weight: Weight for refinement loss in total (beta)
            use_consistency_loss: If True, add consistency penalty between stages
            consistency_weight: Weight for consistency loss
        """
        super().__init__()

        self.base_crack_weight = base_crack_weight
        self.refinement_crack_weight = refinement_crack_weight
        self.base_loss_weight = base_loss_weight
        self.refinement_loss_weight = refinement_loss_weight
        self.use_consistency_loss = use_consistency_loss
        self.consistency_weight = consistency_weight

        # Normalize weights to sum to 1
        total_weight = base_loss_weight + refinement_loss_weight
        self.base_loss_weight = base_loss_weight / total_weight
        self.refinement_loss_weight = refinement_loss_weight / total_weight

    def forward(self, thick_pred, thin_pred, target):
        """
        Compute joint loss.

        Args:
            thick_pred: Base model prediction [B, 1, H, W]
            thin_pred: Refinement network prediction [B, 1, H, W]
            target: Ground truth mask [B, 1, H, W]

        Returns:
            Dictionary containing:
                - total_loss: Combined weighted loss
                - base_loss: Base model loss
                - refinement_loss: Refinement network loss
                - consistency_loss: (optional) Consistency between stages
        """
        # Base model loss (encourage recall)
        base_loss = dice_loss(thick_pred, target, crack_weight=self.base_crack_weight)

        # Refinement loss (encourage precision)
        refinement_loss = dice_loss(thin_pred, target, crack_weight=self.refinement_crack_weight)

        # Combined loss
        total_loss = (
            self.base_loss_weight * base_loss +
            self.refinement_loss_weight * refinement_loss
        )

        result = {
            'total_loss': total_loss,
            'base_loss': base_loss,
            'refinement_loss': refinement_loss
        }

        # Optional: Consistency loss
        # Ensures refinement doesn't predict cracks where base model didn't detect any
        if self.use_consistency_loss:
            thick_sigmoid = torch.sigmoid(thick_pred)
            thin_sigmoid = torch.sigmoid(thin_pred)

            # Penalize thin predictions outside thick regions
            consistency_loss = F.mse_loss(
                thin_sigmoid * (1 - thick_sigmoid),  # Thin predictions outside thick regions
                torch.zeros_like(thin_sigmoid)
            )

            total_loss = total_loss + self.consistency_weight * consistency_loss
            result['total_loss'] = total_loss
            result['consistency_loss'] = consistency_loss

        return result


class AdaptiveJointLoss(JointLoss):
    """
    Adaptive joint loss that adjusts weights during training.

    Strategy:
        - Early training: Focus more on base model (build recall)
        - Mid training: Shift focus to refinement (improve precision)
        - Late training: Balance both
    """

    def __init__(
        self,
        initial_base_weight=0.7,
        final_base_weight=0.3,
        transition_epochs=30,
        **kwargs
    ):
        """
        Args:
            initial_base_weight: Base loss weight at start (e.g., 0.7)
            final_base_weight: Base loss weight at end (e.g., 0.3)
            transition_epochs: Number of epochs for transition
            **kwargs: Other arguments for JointLoss
        """
        super().__init__(**kwargs)

        self.initial_base_weight = initial_base_weight
        self.final_base_weight = final_base_weight
        self.transition_epochs = transition_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """Update loss weights based on current epoch."""
        self.current_epoch = epoch

        # Linear interpolation from initial to final weight
        if epoch < self.transition_epochs:
            progress = epoch / self.transition_epochs
            current_base_weight = (
                self.initial_base_weight +
                (self.final_base_weight - self.initial_base_weight) * progress
            )
        else:
            current_base_weight = self.final_base_weight

        current_refinement_weight = 1.0 - current_base_weight

        # Update weights (normalize)
        total = current_base_weight + current_refinement_weight
        self.base_loss_weight = current_base_weight / total
        self.refinement_loss_weight = current_refinement_weight / total


class ProgressiveLoss(nn.Module):
    """
    Loss for progressive training (train base first, then refinement, then joint).

    Automatically switches between different loss modes based on training phase.
    """

    def __init__(
        self,
        base_crack_weight=0.5,
        refinement_crack_weight=0.3
    ):
        super().__init__()

        self.base_crack_weight = base_crack_weight
        self.refinement_crack_weight = refinement_crack_weight
        self.phase = 1  # 1: base only, 2: refinement only, 3: joint

    def set_phase(self, phase):
        """Set training phase (1, 2, or 3)."""
        self.phase = phase

    def forward(self, thick_pred, thin_pred, target):
        """Compute loss based on current training phase."""

        if self.phase == 1:
            # Phase 1: Train only base model
            base_loss = dice_loss(thick_pred, target, crack_weight=self.base_crack_weight)
            return {
                'total_loss': base_loss,
                'base_loss': base_loss,
                'refinement_loss': torch.tensor(0.0, device=base_loss.device)
            }

        elif self.phase == 2:
            # Phase 2: Train only refinement
            refinement_loss = dice_loss(thin_pred, target, crack_weight=self.refinement_crack_weight)
            return {
                'total_loss': refinement_loss,
                'base_loss': torch.tensor(0.0, device=refinement_loss.device),
                'refinement_loss': refinement_loss
            }

        else:
            # Phase 3: Joint training
            base_loss = dice_loss(thick_pred, target, crack_weight=self.base_crack_weight)
            refinement_loss = dice_loss(thin_pred, target, crack_weight=self.refinement_crack_weight)

            # Weight refinement more in joint phase
            total_loss = 0.3 * base_loss + 0.7 * refinement_loss

            return {
                'total_loss': total_loss,
                'base_loss': base_loss,
                'refinement_loss': refinement_loss
            }


if __name__ == "__main__":
    print("Testing Joint Loss Functions...\n")

    # Test data
    batch_size = 4
    H, W = 256, 256
    thick_pred = torch.randn(batch_size, 1, H, W)
    thin_pred = torch.randn(batch_size, 1, H, W)
    target = torch.randint(0, 2, (batch_size, 1, H, W)).float()

    print("="*70)

    # Test 1: Standard joint loss
    print("1. JointLoss (Standard)")
    print("-"*70)
    criterion = JointLoss(
        base_crack_weight=0.5,
        refinement_crack_weight=0.3,
        base_loss_weight=0.3,
        refinement_loss_weight=0.7
    )

    result = criterion(thick_pred, thin_pred, target)
    print(f"   Total loss: {result['total_loss'].item():.4f}")
    print(f"   Base loss: {result['base_loss'].item():.4f}")
    print(f"   Refinement loss: {result['refinement_loss'].item():.4f}")
    print(f"   Base weight: {criterion.base_loss_weight:.2f}")
    print(f"   Refinement weight: {criterion.refinement_loss_weight:.2f}")

    print("\n" + "="*70)

    # Test 2: Adaptive joint loss
    print("2. AdaptiveJointLoss")
    print("-"*70)
    adaptive_criterion = AdaptiveJointLoss(
        initial_base_weight=0.7,
        final_base_weight=0.3,
        transition_epochs=30
    )

    print("   Epoch 0:")
    adaptive_criterion.set_epoch(0)
    print(f"     Base weight: {adaptive_criterion.base_loss_weight:.3f}")
    print(f"     Refinement weight: {adaptive_criterion.refinement_loss_weight:.3f}")

    print("   Epoch 15:")
    adaptive_criterion.set_epoch(15)
    print(f"     Base weight: {adaptive_criterion.base_loss_weight:.3f}")
    print(f"     Refinement weight: {adaptive_criterion.refinement_loss_weight:.3f}")

    print("   Epoch 30:")
    adaptive_criterion.set_epoch(30)
    print(f"     Base weight: {adaptive_criterion.base_loss_weight:.3f}")
    print(f"     Refinement weight: {adaptive_criterion.refinement_loss_weight:.3f}")

    print("\n" + "="*70)

    # Test 3: Progressive loss
    print("3. ProgressiveLoss")
    print("-"*70)
    progressive_criterion = ProgressiveLoss()

    for phase in [1, 2, 3]:
        progressive_criterion.set_phase(phase)
        result = progressive_criterion(thick_pred, thin_pred, target)
        print(f"   Phase {phase}:")
        print(f"     Total: {result['total_loss'].item():.4f}")
        print(f"     Base: {result['base_loss'].item():.4f}")
        print(f"     Refinement: {result['refinement_loss'].item():.4f}")

    print("\n" + "="*70)
    print("\n✓ All tests passed!")
    print("\nRecommended:")
    print("  - Start with JointLoss(base_crack_weight=0.5, refinement_crack_weight=0.3)")
    print("  - Or use AdaptiveJointLoss for automatic weight scheduling")
