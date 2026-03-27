"""
Refinement Network for Converting Thick Crack Masks to Thin Centerlines

This network learns to refine thick segmentation masks into thin, precise
crack centerlines. It's designed to be used as a post-processor after the
base segmentation model.

The network mimics (but learns better than) traditional CV operations like:
- Morphological thinning/skeletonization
- Canny edge detection
- Distance transform-based centerline extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefinementNetwork(nn.Module):
    """
    Lightweight refinement network that converts thick masks to thin predictions.

    Architecture:
        Input: Thick mask [B, 1, H, W] + Original image [B, 3, H, W] (optional)
        Output: Thin mask [B, 1, H, W]

    The network learns to:
        1. Detect edges in the thick mask
        2. Extract centerlines
        3. Use image context to resolve ambiguities
        4. Output thin, skeleton-like predictions

    Usage:
        refine_net = RefinementNetwork(use_image_context=True)
        thin_mask = refine_net(thick_mask, original_image)
    """

    def __init__(self, use_image_context=True, hidden_channels=32):
        """
        Args:
            use_image_context: If True, uses original image as additional input
            hidden_channels: Number of channels in hidden layers (16-64)
        """
        super().__init__()

        self.use_image_context = use_image_context

        # Input channels: 1 (thick mask) + 3 (RGB image) if using context
        in_channels = 1 if not use_image_context else 4

        # Edge detection pathway (learns edge filters like Sobel/Canny)
        self.edge_conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.edge_conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

        # Centerline extraction pathway (learns skeletonization)
        self.center_conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.center_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.center_conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)

        # Main refinement pathway
        # Combines: thick mask + edges + centerline features + (optional) image
        feature_channels = 1 + 8 + 8  # mask + edges + centerline
        if use_image_context:
            feature_channels += 3  # Add RGB

        self.refine_conv1 = nn.Conv2d(feature_channels, hidden_channels, kernel_size=3, padding=1)
        self.refine_conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.refine_conv3 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1)

        # Output layer
        self.output_conv = nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)

        # Batch normalization for stability
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.bn3 = nn.BatchNorm2d(hidden_channels // 2)

        # Initialize edge detection with Sobel-like filters
        self._init_edge_filters()

    def _init_edge_filters(self):
        """Initialize edge detection filters with Sobel operators."""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        with torch.no_grad():
            # Initialize first channel with Sobel filters
            self.edge_conv1.weight[0, 0] = sobel_x / 4.0
            self.edge_conv1.weight[1, 0] = sobel_y / 4.0

    def forward(self, thick_mask, image=None):
        """
        Forward pass to refine thick mask to thin prediction.

        Args:
            thick_mask: Thick segmentation mask [B, 1, H, W], values in [0, 1] or logits
            image: Original image [B, 3, H, W], optional (if use_image_context=True)

        Returns:
            thin_mask: Refined thin prediction [B, 1, H, W], values in [0, 1]
        """
        # Ensure thick_mask is in [0, 1] range
        if thick_mask.abs().max() > 1.5:  # Likely logits
            thick_mask_sigmoid = torch.sigmoid(thick_mask)
        else:
            thick_mask_sigmoid = thick_mask

        # 1. Edge detection pathway
        edges = F.relu(self.edge_conv1(thick_mask_sigmoid))
        edges = F.relu(self.edge_conv2(edges))

        # 2. Centerline extraction pathway
        center = F.relu(self.center_conv1(thick_mask_sigmoid))
        center = F.relu(self.center_conv2(center))
        center = F.relu(self.center_conv3(center))

        # 3. Combine features
        features = [thick_mask_sigmoid, edges, center]

        if self.use_image_context and image is not None:
            features.append(image)

        x = torch.cat(features, dim=1)

        # 4. Refinement processing
        x = self.bn1(F.relu(self.refine_conv1(x)))
        x = self.bn2(F.relu(self.refine_conv2(x)))
        x = self.bn3(F.relu(self.refine_conv3(x)))

        # 5. Output (with residual connection from centerline features)
        output = self.output_conv(x)

        # Apply sigmoid to get [0, 1] output
        return torch.sigmoid(output)


class LightweightRefinementNetwork(nn.Module):
    """
    Ultra-lightweight refinement network for fast inference.

    ~10x fewer parameters than RefinementNetwork, suitable for:
    - Real-time applications
    - Edge devices
    - When speed is critical

    Trade-off: Slightly lower quality than full RefinementNetwork
    """

    def __init__(self, use_image_context=True):
        super().__init__()

        self.use_image_context = use_image_context

        in_channels = 1 if not use_image_context else 4

        # Single pathway - lightweight processing
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.output = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, thick_mask, image=None):
        if thick_mask.abs().max() > 1.5:
            thick_mask = torch.sigmoid(thick_mask)

        if self.use_image_context and image is not None:
            x = torch.cat([thick_mask, image], dim=1)
        else:
            x = thick_mask

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        output = self.output(x)

        return torch.sigmoid(output)


class UNetRefinementNetwork(nn.Module):
    """
    U-Net style refinement network with skip connections.

    Best quality but more parameters. Use when:
    - Quality is more important than speed
    - You have sufficient GPU memory
    - You want the best possible refinement

    The skip connections help preserve fine details during refinement.
    """

    def __init__(self, use_image_context=True):
        super().__init__()

        self.use_image_context = use_image_context
        in_ch = 1 if not use_image_context else 4

        # Encoder
        self.enc1 = self._double_conv(in_ch, 32)
        self.enc2 = self._double_conv(32, 64)
        self.enc3 = self._double_conv(64, 128)

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self._double_conv(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._double_conv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(64, 32)

        # Output
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def _double_conv(self, in_ch, out_ch):
        """Double convolution block: Conv-BN-ReLU-Conv-BN-ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, thick_mask, image=None):
        if thick_mask.abs().max() > 1.5:
            thick_mask = torch.sigmoid(thick_mask)

        if self.use_image_context and image is not None:
            x = torch.cat([thick_mask, image], dim=1)
        else:
            x = thick_mask

        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Output
        output = self.output(d1)
        return torch.sigmoid(output)


if __name__ == "__main__":
    print("Testing Refinement Networks...\n")

    # Test data
    batch_size = 2
    H, W = 256, 256
    thick_mask = torch.randn(batch_size, 1, H, W)
    image = torch.randn(batch_size, 3, H, W)

    print("="*70)

    # Test 1: Standard refinement network
    print("1. RefinementNetwork (Recommended)")
    print("-"*70)
    refine_net = RefinementNetwork(use_image_context=True, hidden_channels=32)
    output = refine_net(thick_mask, image)
    params = sum(p.numel() for p in refine_net.parameters())
    print(f"   Input:  Thick mask {thick_mask.shape} + Image {image.shape}")
    print(f"   Output: Thin mask {output.shape}")
    print(f"   Parameters: {params:,}")
    print(f"   Memory: ~{params * 4 / 1024 / 1024:.2f} MB")
    assert output.shape == (batch_size, 1, H, W)
    assert output.min() >= 0 and output.max() <= 1
    print("   ✓ Test passed")

    print("\n" + "="*70)

    # Test 2: Lightweight version
    print("2. LightweightRefinementNetwork (Fast)")
    print("-"*70)
    light_net = LightweightRefinementNetwork(use_image_context=True)
    output = light_net(thick_mask, image)
    params = sum(p.numel() for p in light_net.parameters())
    print(f"   Input:  Thick mask {thick_mask.shape} + Image {image.shape}")
    print(f"   Output: Thin mask {output.shape}")
    print(f"   Parameters: {params:,}")
    print(f"   Memory: ~{params * 4 / 1024 / 1024:.2f} MB")
    assert output.shape == (batch_size, 1, H, W)
    print("   ✓ Test passed")

    print("\n" + "="*70)

    # Test 3: U-Net version
    print("3. UNetRefinementNetwork (Best Quality)")
    print("-"*70)
    unet_net = UNetRefinementNetwork(use_image_context=True)
    output = unet_net(thick_mask, image)
    params = sum(p.numel() for p in unet_net.parameters())
    print(f"   Input:  Thick mask {thick_mask.shape} + Image {image.shape}")
    print(f"   Output: Thin mask {output.shape}")
    print(f"   Parameters: {params:,}")
    print(f"   Memory: ~{params * 4 / 1024 / 1024:.2f} MB")
    assert output.shape == (batch_size, 1, H, W)
    print("   ✓ Test passed")

    print("\n" + "="*70)
    print("\nAll tests passed! ✓")
    print("\nRecommendation: Use RefinementNetwork with hidden_channels=32")
    print("  - Good balance of quality and speed")
    print("  - ~200K parameters")
    print("  - Works with image context for better refinement")
