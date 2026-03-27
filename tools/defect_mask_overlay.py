#!/usr/bin/env python3
"""
Script to overlay two different mask types on an image with transparent colors.
Usage: python defect_mask_overlay.py <image_path> <mask1_path> <mask2_path> [output_path]
"""

import os
import sys
from PIL import Image
import numpy as np


def create_dual_mask_overlay(
    image_path: str,
    mask1_path: str,
    mask2_path: str,
    output_path: str = None,
    color1: tuple = (0, 255, 0),      # Green for mask1
    color2: tuple = (0, 0, 255),      # Blue for mask2
    alpha1: float = 0.5,               # Transparency for mask1
    alpha2: float = 0.5,               # Transparency for mask2
    blend_mode: str = "overlay"        # "overlay" or "add" for overlapping regions
):
    """
    Create an overlay of two masks with different colors on an image.

    Args:
        image_path: Path to the original image
        mask1_path: Path to the first mask
        mask2_path: Path to the second mask
        output_path: Path to save the result (if None, will auto-generate)
        color1: RGB color for mask1 (default: green)
        color2: RGB color for mask2 (default: blue)
        alpha1: Transparency for mask1 (0-1, default: 0.5)
        alpha2: Transparency for mask2 (0-1, default: 0.5)
        blend_mode: How to handle overlapping regions ("overlay" or "add")

    Returns:
        Path to the saved output image
    """
    # Load the original image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    # Load masks
    mask1 = Image.open(mask1_path).convert("L")
    mask2 = Image.open(mask2_path).convert("L")

    # Resize masks if they don't match the image size
    if mask1.size != img.size:
        mask1 = mask1.resize(img.size, Image.NEAREST)
    if mask2.size != img.size:
        mask2 = mask2.resize(img.size, Image.NEAREST)

    mask1_array = np.array(mask1) / 255.0  # Normalize to 0-1
    mask2_array = np.array(mask2) / 255.0  # Normalize to 0-1

    # Create colored overlays
    overlay = img_array.copy().astype(float)

    # Apply first mask with color1
    for c in range(3):
        overlay[:, :, c] = (
            img_array[:, :, c] * (1 - alpha1 * mask1_array) +
            color1[c] * alpha1 * mask1_array
        )

    # Apply second mask with color2
    if blend_mode == "overlay":
        # Sequential overlay - mask2 on top of mask1
        for c in range(3):
            overlay[:, :, c] = (
                overlay[:, :, c] * (1 - alpha2 * mask2_array) +
                color2[c] * alpha2 * mask2_array
            )
    elif blend_mode == "add":
        # Additive blending - overlapping regions show mixed color
        for c in range(3):
            overlay[:, :, c] = np.clip(
                overlay[:, :, c] + color2[c] * alpha2 * mask2_array,
                0, 255
            )

    # Convert back to image
    result = Image.fromarray(overlay.astype(np.uint8))

    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(image_path)
        output_path = os.path.join(output_dir, f"{base_name}_overlay.png")

    # Save the result
    result.save(output_path)
    print(f"Saved overlay to: {output_path}")

    return output_path


def process_directory(directory: str, pattern1: str = "_mask.png", pattern2: str = "_mask2.png"):
    """
    Process all images in a directory that have matching mask pairs.

    Args:
        directory: Directory containing images and masks
        pattern1: Suffix pattern for first mask type
        pattern2: Suffix pattern for second mask type
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    # Find all images in directory
    files = os.listdir(directory)

    # Process each potential image
    processed = 0
    for filename in files:
        filepath = os.path.join(directory, filename)
        base_name, ext = os.path.splitext(filename)

        # Skip if not an image or if it's a mask file
        if ext.lower() not in image_extensions or "_mask" in filename or "_overlay" in filename:
            continue

        # Look for corresponding masks
        mask1_path = os.path.join(directory, base_name + pattern1)
        mask2_path = os.path.join(directory, base_name + pattern2)

        if os.path.exists(mask1_path) and os.path.exists(mask2_path):
            print(f"\nProcessing: {filename}")
            create_dual_mask_overlay(filepath, mask1_path, mask2_path)
            processed += 1
        else:
            if not os.path.exists(mask1_path):
                print(f"Skipping {filename}: missing {base_name + pattern1}")
            if not os.path.exists(mask2_path):
                print(f"Skipping {filename}: missing {base_name + pattern2}")

    print(f"\n✓ Processed {processed} images")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image: python defect_mask_overlay.py <image_path> <mask1_path> <mask2_path> [output_path]")
        print("  Batch mode:   python defect_mask_overlay.py <directory>")
        print("\nExample:")
        print("  python defect_mask_overlay.py honeycomb2_result/")
        print("  python defect_mask_overlay.py honeycomb2_result/honeycomb_3.jpg honeycomb2_result/honeycomb_3_mask.png honeycomb2_result/honeycomb_3_mask2.png")
        sys.exit(1)

    # Check if first argument is a directory (batch mode)
    if os.path.isdir(sys.argv[1]):
        process_directory(sys.argv[1])
    # Single image mode
    elif len(sys.argv) >= 4:
        image_path = sys.argv[1]
        mask1_path = sys.argv[2]
        mask2_path = sys.argv[3]
        output_path = sys.argv[4] if len(sys.argv) > 4 else None

        create_dual_mask_overlay(image_path, mask1_path, mask2_path, output_path)
    else:
        print("Error: Please provide either a directory or image_path + mask1_path + mask2_path")
        sys.exit(1)


if __name__ == "__main__":
    main()
