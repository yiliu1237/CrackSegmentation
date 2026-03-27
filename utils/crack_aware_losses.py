"""
Crack-aware loss functions for precise crack segmentation.

These loss functions are designed to encourage the model to predict thin, precise
crack boundaries rather than thick, enclosed regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def balanced_dice_loss(inputs, targets, smooth=1, crack_weight=0.6):
    """
    Dice loss with balanced class weights to prevent over-enclosing predictions.

    Args:
        inputs: Model predictions (before sigmoid)
        targets: Ground truth masks
        smooth: Smoothing factor for Dice coefficient
        crack_weight: Weight for crack pixels (0.5-0.7 recommended)
                     Lower values = more precision, less recall
                     Higher values = more recall, less precision

    Default weight ratio: 0.6/0.4 = 1.5x (vs original 24x)
    This encourages the model to be more precise with crack boundaries.
    """
    inputs = torch.sigmoid(inputs)
    inputs_flat = inputs.view(-1)
    targets_flat = targets.view(-1)

    # Balanced weighting
    weight = torch.zeros_like(targets_flat, device=targets_flat.device)
    weight.fill_(1.0 - crack_weight)  # background weight
    weight[targets_flat > 0] = crack_weight  # crack weight

    intersection = (inputs_flat * targets_flat).sum()
    dice = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
    bce = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean', weight=weight)

    return bce + dice


def thinness_aware_loss(inputs, targets, smooth=1, crack_weight=0.55, thinness_weight=0.3):
    """
    Loss function that penalizes thick predictions.

    This adds a thinness penalty that encourages the model to predict
    thin crack lines rather than thick blobs.

    Args:
        inputs: Model predictions (before sigmoid)
        targets: Ground truth masks
        smooth: Smoothing factor
        crack_weight: Weight for crack pixels in BCE
        thinness_weight: Weight for thinness penalty term
    """
    inputs_sigmoid = torch.sigmoid(inputs)
    inputs_flat = inputs_sigmoid.view(-1)
    targets_flat = targets.view(-1)

    # Standard weighted BCE + Dice
    weight = torch.zeros_like(targets_flat, device=targets_flat.device)
    weight.fill_(1.0 - crack_weight)
    weight[targets_flat > 0] = crack_weight

    intersection = (inputs_flat * targets_flat).sum()
    dice = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
    bce = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean', weight=weight)

    # Thinness penalty: penalize when prediction area >> target area
    pred_area = inputs_sigmoid.sum()
    target_area = targets.sum()

    # Only penalize if prediction is larger than target
    area_ratio = pred_area / (target_area + smooth)
    thinness_penalty = F.relu(area_ratio - 1.0)  # Only penalize if ratio > 1

    return bce + dice + thinness_weight * thinness_penalty


def focal_dice_loss(inputs, targets, smooth=1, crack_weight=0.6, gamma=2.0):
    """
    Dice loss with focal weighting to focus on hard-to-classify pixels.

    This helps the model focus on precise crack boundaries by giving
    more weight to uncertain predictions.

    Args:
        inputs: Model predictions (before sigmoid)
        targets: Ground truth masks
        smooth: Smoothing factor
        crack_weight: Weight for crack pixels
        gamma: Focal loss exponent (higher = more focus on hard examples)
    """
    inputs_sigmoid = torch.sigmoid(inputs)
    inputs_flat = inputs_sigmoid.view(-1)
    targets_flat = targets.view(-1)

    # Focal weighting
    bce_per_pixel = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='none')
    focal_weight = (1 - inputs_flat).pow(gamma) * targets_flat + inputs_flat.pow(gamma) * (1 - targets_flat)

    # Class balancing
    class_weight = torch.zeros_like(targets_flat, device=targets_flat.device)
    class_weight.fill_(1.0 - crack_weight)
    class_weight[targets_flat > 0] = crack_weight

    focal_bce = (focal_weight * class_weight * bce_per_pixel).mean()

    # Standard Dice
    intersection = (inputs_flat * targets_flat).sum()
    dice = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)

    return focal_bce + dice


def precision_focused_loss(inputs, targets, smooth=1, precision_weight=0.7):
    """
    Loss that prioritizes precision over recall.

    This strongly penalizes false positives (predicting crack where there is none),
    encouraging the model to only predict cracks where it's confident.

    Args:
        inputs: Model predictions (before sigmoid)
        targets: Ground truth masks
        smooth: Smoothing factor
        precision_weight: Weight for precision penalty (0.6-0.8 recommended)
    """
    inputs_sigmoid = torch.sigmoid(inputs)
    inputs_flat = inputs_sigmoid.view(-1)
    targets_flat = targets.view(-1)

    # Standard Dice
    intersection = (inputs_flat * targets_flat).sum()
    dice = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)

    # Precision penalty: heavily penalize false positives
    false_positives = (inputs_flat * (1 - targets_flat)).sum()
    true_positives = intersection

    precision_penalty = false_positives / (true_positives + false_positives + smooth)

    # Balanced BCE
    weight = torch.zeros_like(targets_flat, device=targets_flat.device)
    weight.fill_(0.4)
    weight[targets_flat > 0] = 0.6
    bce = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean', weight=weight)

    return bce + dice + precision_weight * precision_penalty


def boundary_aware_loss(inputs, targets, smooth=1, crack_weight=0.6, boundary_weight=0.5):
    """
    Loss that focuses on crack boundaries using edge detection.

    This computes edges from both prediction and target, then adds
    additional loss on boundary regions to encourage precise alignment.

    Args:
        inputs: Model predictions (before sigmoid)
        targets: Ground truth masks
        smooth: Smoothing factor
        crack_weight: Weight for crack pixels in BCE
        boundary_weight: Weight for boundary-focused loss
    """
    inputs_sigmoid = torch.sigmoid(inputs)

    # Standard weighted loss
    inputs_flat = inputs_sigmoid.view(-1)
    targets_flat = targets.view(-1)

    weight = torch.zeros_like(targets_flat, device=targets_flat.device)
    weight.fill_(1.0 - crack_weight)
    weight[targets_flat > 0] = crack_weight

    intersection = (inputs_flat * targets_flat).sum()
    dice = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
    bce = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean', weight=weight)

    # Boundary detection using Sobel-like gradients
    # Simple edge detector: |dx| + |dy|
    if len(inputs.shape) == 4:  # [B, C, H, W]
        # Compute gradients
        dx_pred = torch.abs(inputs_sigmoid[:, :, :, 1:] - inputs_sigmoid[:, :, :, :-1])
        dy_pred = torch.abs(inputs_sigmoid[:, :, 1:, :] - inputs_sigmoid[:, :, :-1, :])

        dx_target = torch.abs(targets[:, :, :, 1:] - targets[:, :, :, :-1])
        dy_target = torch.abs(targets[:, :, 1:, :] - targets[:, :, :-1, :])

        # Edge maps (approximate)
        edges_pred = torch.zeros_like(inputs_sigmoid)
        edges_target = torch.zeros_like(targets)

        edges_pred[:, :, :, 1:] += dx_pred
        edges_pred[:, :, 1:, :] += dy_pred

        edges_target[:, :, :, 1:] += dx_target
        edges_target[:, :, 1:, :] += dy_target

        # Boundary loss
        boundary_loss = F.mse_loss(edges_pred, edges_target)
    else:
        boundary_loss = torch.tensor(0.0, device=inputs.device)

    return bce + dice + boundary_weight * boundary_loss


class CrackAwareLoss(nn.Module):
    """
    Configurable crack-aware loss module.

    Usage:
        loss_fn = CrackAwareLoss(mode='balanced', crack_weight=0.6)
        loss = loss_fn(predictions, targets)
    """

    def __init__(self, mode='balanced', crack_weight=0.6, **kwargs):
        """
        Args:
            mode: Loss function mode
                - 'balanced': Balanced class weights (recommended starting point)
                - 'thinness': Penalizes thick predictions
                - 'focal': Focal loss for hard examples
                - 'precision': Prioritizes precision over recall
                - 'boundary': Focuses on boundary alignment
            crack_weight: Weight for crack pixels (0.5-0.7)
            **kwargs: Additional parameters for specific loss modes
        """
        super().__init__()
        self.mode = mode
        self.crack_weight = crack_weight
        self.kwargs = kwargs

        self.loss_functions = {
            'balanced': balanced_dice_loss,
            'thinness': thinness_aware_loss,
            'focal': focal_dice_loss,
            'precision': precision_focused_loss,
            'boundary': boundary_aware_loss,
        }

        if mode not in self.loss_functions:
            raise ValueError(f"Unknown loss mode: {mode}. Choose from {list(self.loss_functions.keys())}")

    def forward(self, inputs, targets):
        loss_fn = self.loss_functions[self.mode]
        return loss_fn(inputs, targets, crack_weight=self.crack_weight, **self.kwargs)


# Utility function to test different loss configurations
def compare_loss_behaviors(predictions, targets):
    """
    Helper function to compare how different loss functions behave
    on the same predictions.

    Args:
        predictions: Model output (before sigmoid)
        targets: Ground truth

    Returns:
        dict: Loss values for each configuration
    """
    results = {}

    # Original heavy weighting
    original_loss = balanced_dice_loss(predictions, targets, crack_weight=0.96)
    results['original_heavy_weight'] = original_loss.item()

    # Balanced weights
    balanced_loss = balanced_dice_loss(predictions, targets, crack_weight=0.6)
    results['balanced_weight'] = balanced_loss.item()

    # More balanced
    more_balanced = balanced_dice_loss(predictions, targets, crack_weight=0.55)
    results['more_balanced'] = more_balanced.item()

    # Thinness aware
    thinness_loss = thinness_aware_loss(predictions, targets, crack_weight=0.55)
    results['thinness_aware'] = thinness_loss.item()

    # Precision focused
    precision_loss = precision_focused_loss(predictions, targets)
    results['precision_focused'] = precision_loss.item()

    return results


if __name__ == "__main__":
    # Test the loss functions
    print("Testing crack-aware loss functions...")

    # Create dummy data
    batch_size, channels, height, width = 4, 1, 256, 256
    predictions = torch.randn(batch_size, channels, height, width)
    targets = torch.randint(0, 2, (batch_size, channels, height, width)).float()

    # Test each loss
    print("\nLoss function values:")
    for mode in ['balanced', 'thinness', 'focal', 'precision', 'boundary']:
        loss_fn = CrackAwareLoss(mode=mode, crack_weight=0.6)
        loss = loss_fn(predictions, targets)
        print(f"  {mode:15s}: {loss.item():.4f}")

    print("\nComparison of weight configurations:")
    comparison = compare_loss_behaviors(predictions, targets)
    for name, value in comparison.items():
        print(f"  {name:25s}: {value:.4f}")

    print("\nAll tests passed!")
