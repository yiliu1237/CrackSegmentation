#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Mask Area Analysis (with connected component separation)

Analyzes each connected component in a mask separately and calculates:
- Actual mask area (pixel count) for each component
- Bounding box dimensions for each component
- Bbox density and aspect ratio for each component

Usage:
    python defect_mask_analysis.py --input <mask_path> --output <output_dir>
"""

import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure
import json


class ConnectedComponentMaskAnalyzer:
    """Analyze area metrics for mask with separate connected components."""

    def __init__(self, min_component_size=100):
        """
        Initialize analyzer.

        Args:
            min_component_size: Minimum number of pixels for a component to be analyzed (default: 100)
        """
        self.min_component_size = min_component_size

    def load_mask(self, mask_path):
        """Load binary mask from file."""
        mask = np.array(Image.open(mask_path).convert('L'))
        return mask

    def find_connected_components(self, mask, threshold=127):
        """
        Find separate connected components in the mask.

        Args:
            mask: Grayscale mask image
            threshold: Binarization threshold (default: 127)

        Returns:
            labeled_mask: Labeled image where each component has a unique integer
            num_components: Number of components found
        """
        binary = mask > threshold
        labeled_mask = measure.label(binary, connectivity=2)  # 8-connectivity
        num_components = labeled_mask.max()

        return labeled_mask, num_components

    def calculate_component_metrics(self, coords, component_id, mask_shape):
        """
        Calculate area-based metrics for a single connected component.

        Args:
            coords: (N, 2) array of pixel coordinates for this component
            component_id: ID of this component
            mask_shape: Shape of the full mask

        Returns:
            Dictionary of metrics for this component
        """
        metrics = {'component_id': component_id}

        if len(coords) == 0:
            return metrics

        # 1. Actual mask area (pixel count)
        actual_area = len(coords)
        metrics['actual_mask_area'] = int(actual_area)

        # 2. Bounding box metrics
        y_coords, x_coords = coords[:, 0], coords[:, 1]
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        x_min, x_max = int(x_coords.min()), int(x_coords.max())

        bbox_width = x_max - x_min + 1
        bbox_height = y_max - y_min + 1
        bbox_area = bbox_width * bbox_height

        metrics['bbox_width'] = int(bbox_width)
        metrics['bbox_height'] = int(bbox_height)
        metrics['bbox_area'] = int(bbox_area)
        metrics['bbox_position'] = {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
        }
        metrics['bbox_density'] = float(actual_area / bbox_area)

        # 3. Aspect ratio
        metrics['aspect_ratio'] = float(bbox_width / bbox_height)

        return metrics

    def analyze(self, mask_path):
        """
        Perform complete analysis on the mask.

        Args:
            mask_path: Path to mask image

        Returns:
            Dictionary containing all calculated metrics
        """
        print(f"[INFO] Loading mask: {mask_path}")
        mask = self.load_mask(mask_path)

        # Find connected components
        print("[INFO] Finding connected components...")
        labeled_mask, num_components = self.find_connected_components(mask)
        print(f"[INFO] Found {num_components} connected components")

        if num_components == 0:
            print("[ERROR] No components detected in mask")
            return None

        # Analyze each component
        components_metrics = []
        all_coords_by_component = []

        for comp_id in range(1, num_components + 1):
            # Extract coordinates for this component
            coords = np.column_stack(np.where(labeled_mask == comp_id))

            # Skip small components
            if len(coords) < self.min_component_size:
                print(f"[INFO] Skipping component {comp_id} (only {len(coords)} pixels, < {self.min_component_size})")
                continue

            print(f"[INFO] Analyzing component {comp_id}: {len(coords):,} pixels")

            metrics = self.calculate_component_metrics(coords, comp_id, mask.shape)
            components_metrics.append(metrics)
            all_coords_by_component.append(coords)

        if len(components_metrics) == 0:
            print("[ERROR] No components large enough to analyze")
            return None

        # Calculate total statistics
        total_pixels = sum([m['actual_mask_area'] for m in components_metrics])
        total_coverage = 100 * total_pixels / (mask.shape[0] * mask.shape[1])

        # Prepare results
        results = {
            'num_components': len(components_metrics),
            'total_components_found': num_components,
            'min_component_size': self.min_component_size,
            'image_dimensions': {
                'width': int(mask.shape[1]),
                'height': int(mask.shape[0]),
                'total_area': int(mask.shape[0] * mask.shape[1]),
            },
            'total_mask_area': int(total_pixels),
            'total_coverage_percentage': float(total_coverage),
            'components': components_metrics,
            '_visualization_data': {
                'labeled_mask': labeled_mask,
                'mask': mask,
                'coords_by_component': all_coords_by_component,
            }
        }

        return results

    def print_results(self, results):
        """Print analysis results in a formatted way."""
        if results is None:
            return

        print("\n" + "="*70)
        print("CONNECTED COMPONENT MASK ANALYSIS RESULTS")
        print("="*70)

        # Image dimensions
        print("\n--- Image Dimensions ---")
        img_dim = results['image_dimensions']
        print(f"Width × Height:         {img_dim['width']} × {img_dim['height']} pixels")
        print(f"Total image area:       {img_dim['total_area']:,} px²")

        # Summary
        print("\n--- Summary ---")
        print(f"Total components found: {results['total_components_found']}")
        print(f"Components analyzed:    {results['num_components']} (min size: {results['min_component_size']} px)")
        print(f"Total mask area:        {results['total_mask_area']:,} px²")
        print(f"Total coverage:         {results['total_coverage_percentage']:.2f}% of image")

        # Component details
        for i, comp in enumerate(results['components'], 1):
            print(f"\n--- Component {comp['component_id']} ---")
            print(f"Actual mask area:       {comp['actual_mask_area']:,} px²")

            print(f"Bounding box:           {comp['bbox_width']} × {comp['bbox_height']} pixels")
            print(f"Bounding box area:      {comp['bbox_area']:,} px²")
            print(f"Bounding box position:  ({comp['bbox_position']['x_min']}, {comp['bbox_position']['y_min']}) to "
                  f"({comp['bbox_position']['x_max']}, {comp['bbox_position']['y_max']})")
            print(f"Bbox density:           {comp['bbox_density']:.4f}")
            print(f"Aspect ratio (W/H):     {comp['aspect_ratio']:.2f}")

        print("\n" + "="*70)

    def visualize(self, mask_path, results, output_path=None, original_image_path=None, bbox_min_area=0):
        """Create visualization of the analysis results."""
        if results is None or '_visualization_data' not in results:
            return

        viz_data = results['_visualization_data']
        labeled_mask = viz_data['labeled_mask']
        mask = viz_data['mask']
        coords_by_component = viz_data['coords_by_component']

        # Create output subfolder for visualizations
        if output_path:
            viz_dir = Path(output_path).parent / 'visualizations'
            viz_dir.mkdir(parents=True, exist_ok=True)
        else:
            viz_dir = Path('.')

        # === 1. Original Mask ===
        fig1, ax1 = plt.subplots(figsize=(12, 10))
        ax1.imshow(mask, cmap='gray')
        ax1.set_title(f'Original Mask\n({results["total_mask_area"]:,} total pixels, {results["num_components"]} components)',
                     fontsize=14, fontweight='bold')
        ax1.axis('off')
        fig1.savefig(viz_dir / '1_original_mask.png', dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f"[INFO] Saved: {viz_dir / '1_original_mask.png'}")

        # === 2. Labeled Components (Color-coded) ===
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        # Use a colormap to show different components
        cmap = plt.cm.get_cmap('tab20', results['num_components'])
        ax2.imshow(labeled_mask, cmap=cmap, interpolation='nearest')
        ax2.set_title(f'Connected Components\n({results["num_components"]} components shown)',
                     fontsize=14, fontweight='bold')
        ax2.axis('off')

        # Add a simple legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=cmap(i), label=f'Component {results["components"][i]["component_id"]}')
                          for i in range(min(10, len(results['components'])))]  # Show first 10
        if len(results['components']) > 10:
            legend_elements.append(Patch(facecolor='gray', label='...'))
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)

        fig2.savefig(viz_dir / '2_labeled_components.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"[INFO] Saved: {viz_dir / '2_labeled_components.png'}")

        # === 3. Components with Bounding Boxes ===
        fig3, ax3 = plt.subplots(figsize=(14, 12))
        ax3.set_facecolor('white')

        # Assign colors to each component
        colors = plt.cm.get_cmap('tab20', results['num_components'])

        # Draw all components
        for i, (coords, comp_metrics) in enumerate(zip(coords_by_component, results['components'])):
            color = colors(i)

            # Plot component pixels
            ax3.scatter(coords[:, 1], coords[:, 0], c=[color], s=1, alpha=0.4,
                       label=f"Comp {comp_metrics['component_id']} ({comp_metrics['actual_mask_area']:,} px)")

            # Draw bounding box
            bbox_pos = comp_metrics['bbox_position']
            from matplotlib.patches import Rectangle
            rect = Rectangle((bbox_pos['x_min'], bbox_pos['y_min']),
                           comp_metrics['bbox_width'], comp_metrics['bbox_height'],
                           linewidth=2, edgecolor=color, facecolor='none', linestyle='-')
            ax3.add_patch(rect)

        ax3.set_title(f'Components with Bounding Boxes\n'
                     f'({results["num_components"]} components)',
                     fontsize=14, fontweight='bold')
        ax3.invert_yaxis()
        if results['num_components'] <= 10:  # Only show legend if not too many components
            ax3.legend(loc='upper right', fontsize=8, ncol=1)
        ax3.set_xlabel('X (pixels)', fontsize=12)
        ax3.set_ylabel('Y (pixels)', fontsize=12)
        fig3.savefig(viz_dir / '3_components_with_bbox.png', dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f"[INFO] Saved: {viz_dir / '3_components_with_bbox.png'}")

        # === 4. Original Image with Overlay (if available) ===
        if original_image_path is None:
            original_image_path = self._find_original_image(mask_path)
        else:
            original_image_path = Path(original_image_path)

        if original_image_path and original_image_path.exists():
            print(f"[INFO] Found original image: {original_image_path}")
            original_img = np.array(Image.open(original_image_path).convert('RGB'))

            fig4, ax4 = plt.subplots(figsize=(16, 12))
            ax4.imshow(original_img)

            # Create transparent overlay for all components
            H_img, W_img = original_img.shape[:2]
            overlay = np.zeros((H_img, W_img, 4))

            for i, coords in enumerate(coords_by_component):
                color = colors(i)
                # Set color for this component
                overlay[coords[:, 0], coords[:, 1], :3] = color[:3]
                overlay[coords[:, 0], coords[:, 1], 3] = 0.4  # Alpha

            ax4.imshow(overlay, interpolation='none')

            # Draw bounding boxes for each component
            from matplotlib.patches import Rectangle
            for i, (coords, comp_metrics) in enumerate(zip(coords_by_component, results['components'])):
                color = colors(i)

                bbox_pos = comp_metrics['bbox_position']
                if int(comp_metrics.get('actual_mask_area', 0)) < int(bbox_min_area):
                    continue
                rect = Rectangle((bbox_pos['x_min'], bbox_pos['y_min']),
                               comp_metrics['bbox_width'], comp_metrics['bbox_height'],
                               linewidth=3, edgecolor=color, facecolor='none',
                               linestyle='-', label=f"Component {comp_metrics['component_id']}")
                ax4.add_patch(rect)

            # Add statistics text box
            stats_lines = [
                "MASK COMPONENT STATISTICS",
                "",
                f"Total Components: {results['num_components']}",
                f"Total Mask Area: {results['total_mask_area']:,} px²",
                f"Coverage: {results['total_coverage_percentage']:.2f}%",
                "",
            ]

            # Add top 3 components by size
            sorted_comps = sorted(results['components'], key=lambda x: x['actual_mask_area'], reverse=True)
            for i, comp in enumerate(sorted_comps[:3], 1):
                density = comp.get('bbox_density', 0)
                stats_lines.append(f"Top {i}: {comp['actual_mask_area']:,} px², Density: {density:.2f}")

            stats_text = '\n'.join(stats_lines)

            ax4.text(0.02, 0.98, stats_text,
                    transform=ax4.transAxes,
                    fontsize=13,
                    fontweight='bold',
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=1',
                             facecolor='white',
                             alpha=0.95,
                             edgecolor='black',
                             linewidth=2),
                    family='monospace')

            if results['num_components'] <= 10:
                ax4.legend(loc='lower right', fontsize=10, framealpha=0.9, ncol=2)

            ax4.set_title('Component Overlay on Feature 2',
                         fontsize=16, fontweight='bold', pad=20)
            ax4.axis('off')

            fig4.savefig(viz_dir / '4_original_with_overlay.png', dpi=150, bbox_inches='tight')
            plt.close(fig4)
            print(f"[INFO] Saved: {viz_dir / '4_original_with_overlay.png'}")

            # === 5. Original Image with Mask1 + Mask2 Overlay ===
            mask2_path = self._find_mask2(mask_path)
            if mask2_path and mask2_path.exists():
                print(f"[INFO] Found mask2: {mask2_path}")
                mask2 = self.load_mask(mask2_path)
                # Extract mask2 coordinates
                binary_mask2 = mask2 > 127
                coords_mask2 = np.column_stack(np.where(binary_mask2))

                # Create figure with two subplots: image on left, stats on right
                fig5 = plt.figure(figsize=(20, 12))
                gs = fig5.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
                ax_img = fig5.add_subplot(gs[0])
                ax_stats = fig5.add_subplot(gs[1])

                # Plot the image
                ax_img.imshow(original_img)

                # Create transparent overlay for mask1 components (multi-colored, same as viz 4)
                H_img, W_img = original_img.shape[:2]
                overlay_mask1 = np.zeros((H_img, W_img, 4))

                for i, coords in enumerate(coords_by_component):
                    color = colors(i)
                    # Set color for this component
                    overlay_mask1[coords[:, 0], coords[:, 1], :3] = color[:3]
                    overlay_mask1[coords[:, 0], coords[:, 1], 3] = 0.4  # Alpha

                ax_img.imshow(overlay_mask1, interpolation='none')

                # Create transparent overlay for mask2 (blue)
                mask2_area = 0
                if len(coords_mask2) > 0:
                    mask2_area = len(coords_mask2)
                    overlay_mask2 = np.zeros((H_img, W_img, 4))
                    overlay_mask2[coords_mask2[:, 0], coords_mask2[:, 1], 2] = 1.0  # Blue channel
                    overlay_mask2[coords_mask2[:, 0], coords_mask2[:, 1], 3] = 0.4  # Alpha
                    ax_img.imshow(overlay_mask2, interpolation='none')

                # Draw bounding boxes for mask1 components (same colors as components)
                from matplotlib.patches import Rectangle
                for i, (coords, comp_metrics) in enumerate(zip(coords_by_component, results['components'])):
                    color = colors(i)
                    bbox_pos = comp_metrics['bbox_position']
                    if int(comp_metrics.get('actual_mask_area', 0)) < int(bbox_min_area):
                        continue
                    rect = Rectangle((bbox_pos['x_min'], bbox_pos['y_min']),
                                   comp_metrics['bbox_width'], comp_metrics['bbox_height'],
                                   linewidth=3, edgecolor=color, facecolor='none',
                                   linestyle='-')
                    ax_img.add_patch(rect)

                ax_img.set_title('Honeycomb Detection and Analysis',
                                fontsize=16, fontweight='bold', pad=20)
                ax_img.axis('off')

                # Add statistics in the right panel
                ax_stats.axis('off')

                stats_lines = [
                    "ANALYSIS RESULTS",
                    "=" * 35,
                    "",
                    "Green Regions (Honeycomb):",
                    f"  Total Components: {results['num_components']}",
                    f"  Total Area: {results['total_mask_area']:,} px²",
                    f"  Coverage: {results['total_coverage_percentage']:.2f}%",
                    "",
                    "  Component Areas:",
                ]

                # Show all components sorted by size
                sorted_comps = sorted(results['components'], key=lambda x: x['actual_mask_area'], reverse=True)
                for i, comp in enumerate(sorted_comps, 1):
                    stats_lines.append(f"    #{i}: {comp['actual_mask_area']:,} px²")

                # Add mask2 analysis
                mask2_coverage = 100 * mask2_area / (H_img * W_img) if (H_img * W_img) > 0 else 0
                stats_lines.extend([
                    "",
                    "Blue Regions (Additional):",
                    f"  Total Area: {mask2_area:,} px²",
                    f"  Coverage: {mask2_coverage:.2f}%",
                ])

                stats_text = '\n'.join(stats_lines)

                ax_stats.text(0.05, 0.95, stats_text,
                            transform=ax_stats.transAxes,
                            fontsize=11,
                            fontweight='normal',
                            verticalalignment='top',
                            horizontalalignment='left',
                            family='monospace')

                fig5.savefig(viz_dir / '5_dual_mask_overlay.png', dpi=150, bbox_inches='tight')
                plt.close(fig5)
                print(f"[INFO] Saved: {viz_dir / '5_dual_mask_overlay.png'}")
            else:
                print(f"[INFO] Mask2 not found, skipping dual mask overlay")

        else:
            print(f"[INFO] Original image not found, skipping overlay visualization")

        print(f"[INFO] All visualizations saved to: {viz_dir}")

    def _find_original_image(self, mask_path):
        """Try to find the corresponding original image for a mask."""
        mask_path = Path(mask_path)
        base_dir = mask_path.parent

        # Try different naming patterns
        patterns = [
            # For honeycomb2_result structure
            mask_path.stem.replace('_mask', '') + '.jpg',
            mask_path.stem.replace('_mask', '') + '.jpeg',
            mask_path.stem.replace('_mask', '') + '.png',
            # With _original suffix
            mask_path.stem.replace('_mask', '_original') + '.jpg',
            mask_path.stem + '_original.jpg',
        ]

        for pattern in patterns:
            original_path = base_dir / pattern
            if original_path.exists():
                return original_path

        return None

    def _find_mask2(self, mask_path):
        """Try to find the corresponding mask2 for a mask."""
        mask_path = Path(mask_path)
        base_dir = mask_path.parent

        # Replace _mask with _mask2
        mask2_name = mask_path.stem.replace('_mask', '_mask2') + mask_path.suffix
        mask2_path = base_dir / mask2_name

        if mask2_path.exists():
            return mask2_path

        return None

    def save_results(self, results, output_path):
        """Save results to JSON file."""
        if results is None:
            return

        # Remove non-serializable visualization data
        results_copy = results.copy()
        if '_visualization_data' in results_copy:
            del results_copy['_visualization_data']

        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(output_path, 'w') as f:
            json.dump(results_copy, f, indent=2, cls=NumpyEncoder)

        print(f"[INFO] Results saved to: {output_path}")


SimpleMaskAnalyzer = ConnectedComponentMaskAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Analyze mask with connected component separation')
    parser.add_argument('--input', '-i', required=True, help='Input mask image path')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--min-size', type=int, default=100, help='Minimum component size in pixels (default: 100)')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')

    args = parser.parse_args()

    # Create analyzer
    analyzer = ConnectedComponentMaskAnalyzer(min_component_size=args.min_size)

    # Run analysis
    results = analyzer.analyze(args.input)

    if results is None:
        print("[ERROR] Analysis failed")
        return

    # Print results
    analyzer.print_results(results)

    # Save results if output directory specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        json_path = output_dir / 'mask_metrics.json'
        analyzer.save_results(results, json_path)

        # Save visualizations
        if not args.no_viz:
            dummy_path = output_dir / 'dummy.png'
            analyzer.visualize(args.input, results, dummy_path)
    else:
        # If no output dir specified, create default one next to input
        if not args.no_viz:
            output_dir = Path(args.input).parent / f"analysis_{Path(args.input).stem}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save JSON results
            json_path = output_dir / 'mask_metrics.json'
            analyzer.save_results(results, json_path)

            # Save visualizations
            dummy_path = output_dir / 'dummy.png'
            analyzer.visualize(args.input, results, dummy_path)


if __name__ == '__main__':
    main()

