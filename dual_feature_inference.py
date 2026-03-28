from __future__ import division

import argparse
import json
import os
import sys
import tempfile
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from dataloaders import ImgToTensor
from model import CrackModelDinoV3
import utils

# Make local imports robust when script is run from different working directories.
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
TOOLS_DIR = CURRENT_DIR / "tools"
if TOOLS_DIR.exists() and str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools.defect_mask_analysis import ConnectedComponentMaskAnalyzer
from tools.defect_mask_overlay import create_dual_mask_overlay

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


@dataclass
class DualInferenceConfig:
    task: str = "honeycomb_dual_feature"
    images_dir: str = ""
    checkpoint1: str = ""
    checkpoint2: str = ""
    save_result: str = ""
    save_result_root: str = "./test_result"
    dinov3_arch: str = "convnext_base"
    dinov3_ckpt: str = "./pretrained_checkpoint/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"
    bin_thresh1: float = 0.5
    bin_thresh2: float = 0.5
    patch_size: int = 512
    patch_stride_ratio: float = 0.66
    infer_batch_size: int = 16
    min_component_size: int = 100
    panel_bbox_min_size: int = 300
    output_name_mode: str = "original"  # "original" or "index"
    output_prefix: str = ""


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-model defect inference pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument("--image-dir", type=str, default=None, help="Input image directory.")
    parser.add_argument("--checkpoint1", type=str, default=None, help="Checkpoint path for model 1.")
    parser.add_argument("--checkpoint2", type=str, default=None, help="Checkpoint path for model 2.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--bin-thresh1", type=float, default=None, help="Binary threshold for model 1 mask.")
    parser.add_argument("--bin-thresh2", type=float, default=None, help="Binary threshold for model 2 mask.")
    parser.add_argument("--min-component-size", type=int, default=None, help="Minimum component size in analysis.")
    parser.add_argument(
        "--panel-bbox-min-size",
        type=int,
        default=None,
        help="Minimum component area (pixels) to show bbox/component in analysis panel. "
             "Does not change saved masks.",
    )
    parser.add_argument("--output-prefix", type=str, default=None, help="Prefix for output filenames.")
    parser.add_argument(
        "--output-name-mode",
        type=str,
        choices=["original", "index"],
        default=None,
        help="Naming mode: original image stem or running index.",
    )
    return parser.parse_args()


def load_config(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("dual_inference", data.get("inference", data))


def build_config(args):
    cfg = DualInferenceConfig()
    merged = asdict(cfg)
    merged.update(load_config(args.config))

    if args.image_dir is not None:
        merged["images_dir"] = args.image_dir
    if args.checkpoint1 is not None:
        merged["checkpoint1"] = args.checkpoint1
    if args.checkpoint2 is not None:
        merged["checkpoint2"] = args.checkpoint2
    if args.output_dir is not None:
        merged["save_result"] = args.output_dir
    if args.bin_thresh1 is not None:
        merged["bin_thresh1"] = args.bin_thresh1
    if args.bin_thresh2 is not None:
        merged["bin_thresh2"] = args.bin_thresh2
    if args.min_component_size is not None:
        merged["min_component_size"] = args.min_component_size
    if args.panel_bbox_min_size is not None:
        merged["panel_bbox_min_size"] = args.panel_bbox_min_size
    if args.output_prefix is not None:
        merged["output_prefix"] = args.output_prefix
    if args.output_name_mode is not None:
        merged["output_name_mode"] = args.output_name_mode

    if not merged.get("images_dir"):
        raise ValueError("images_dir is required. Set config or pass --image-dir.")
    if not merged.get("checkpoint1"):
        raise ValueError("checkpoint1 is required. Set config or pass --checkpoint1.")
    if not merged.get("checkpoint2"):
        raise ValueError("checkpoint2 is required. Set config or pass --checkpoint2.")
    if not merged.get("save_result"):
        merged["save_result"] = os.path.join(merged["save_result_root"], merged["task"])

    return DualInferenceConfig(**merged)


def list_images(images_dir):
    exts = ("*.JPG", "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for ext in exts:
        files.extend(Path(images_dir).glob(ext))
    return sorted(set(files))


def serialize_mask_metrics(metrics):
    if metrics is None:
        return {
            "num_components": 0,
            "total_components_found": 0,
            "min_component_size": 0,
            "image_dimensions": {"width": 0, "height": 0, "total_area": 0},
            "total_mask_area": 0,
            "total_coverage_percentage": 0.0,
            "components": [],
        }
    data = dict(metrics)
    data.pop("_visualization_data", None)
    return to_builtin_types(data)


def to_builtin_types(obj):
    """Recursively convert numpy/scalar containers to JSON-safe builtin Python types."""
    if isinstance(obj, dict):
        return {str(k): to_builtin_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin_types(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def analyze_single_mask(mask_path, analyzer):
    mask = analyzer.load_mask(mask_path)
    labeled_mask, num_components = analyzer.find_connected_components(mask)

    components_metrics = []
    for comp_id in range(1, num_components + 1):
        coords = np.column_stack(np.where(labeled_mask == comp_id))
        if len(coords) < analyzer.min_component_size:
            continue
        components_metrics.append(analyzer.calculate_component_metrics(coords, comp_id, mask.shape))

    total_pixels = sum(m["actual_mask_area"] for m in components_metrics)
    total_coverage = 100 * total_pixels / (mask.shape[0] * mask.shape[1]) if mask.size > 0 else 0.0

    return {
        "num_components": len(components_metrics),
        "total_components_found": int(num_components),
        "min_component_size": int(analyzer.min_component_size),
        "image_dimensions": {
            "width": int(mask.shape[1]),
            "height": int(mask.shape[0]),
            "total_area": int(mask.shape[0] * mask.shape[1]),
        },
        "total_mask_area": int(total_pixels),
        "total_coverage_percentage": float(total_coverage),
        "components": components_metrics,
    }


def create_single_mask_overlay(image_rgb, mask_u8, color=(0, 255, 0), alpha=0.45):
    overlay = image_rgb.copy().astype(np.float32)
    mask_bin = (mask_u8 > 127).astype(np.float32)
    for c in range(3):
        overlay[:, :, c] = (
            image_rgb[:, :, c].astype(np.float32) * (1.0 - alpha * mask_bin) +
            float(color[c]) * alpha * mask_bin
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


class DualFeatureInference:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_to_tensor = ImgToTensor()
        self.analyzer = ConnectedComponentMaskAnalyzer(min_component_size=config.min_component_size)
        self.model1, self.model2 = self.init_models()

    def init_models(self):
        model1 = CrackModelDinoV3(
            dinov3_arch=self.config.dinov3_arch,
            dinov3_ckpt=self.config.dinov3_ckpt,
        ).to(self.device)
        model2 = CrackModelDinoV3(
            dinov3_arch=self.config.dinov3_arch,
            dinov3_ckpt=self.config.dinov3_ckpt,
        ).to(self.device)

        state_dict_1 = torch.load(self.config.checkpoint1, map_location="cpu")
        state_dict_2 = torch.load(self.config.checkpoint2, map_location="cpu")
        model1.load_state_dict(state_dict_1, strict=True)
        model2.load_state_dict(state_dict_2, strict=True)
        model1.eval()
        model2.eval()
        return model1, model2

    def infer_single_model(self, model, image_rgb):
        h, w = image_rgb.shape[:2]
        image_tensor = self.img_to_tensor(image_rgb).squeeze(0)
        patches, patch_locs = utils.get_img_patches_adjusted(
            image_tensor, self.config.patch_size, stride_ratio=self.config.patch_stride_ratio
        )
        patches = patches.to(self.device, non_blocking=True)

        patch_list = torch.split(patches, self.config.infer_batch_size)
        model_results = []
        with torch.no_grad():
            for each in patch_list:
                model_results.extend(torch.sigmoid(model(each)).cpu())
        pred_mask = torch.stack(model_results, 0)
        full_mask = utils.merge_pred_patches_adjusted(image_tensor, pred_mask, patch_locs)
        full_mask = full_mask.detach().numpy()
        full_mask = cv2.resize(full_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return np.clip(full_mask, 0.0, 1.0)

    def get_output_stem(self, image_path, index):
        if self.config.output_name_mode == "index":
            base = str(index)
        else:
            base = Path(image_path).stem
        if self.config.output_prefix:
            return f"{self.config.output_prefix}{base}"
        return base

    def run(self):
        os.makedirs(self.config.save_result, exist_ok=True)
        image_paths = list_images(self.config.images_dir)
        if not image_paths:
            raise ValueError(
                f"No images found in '{self.config.images_dir}'. Supported: JPG/JPEG/PNG/BMP/TIF/TIFF."
            )

        print(f"[INFO] Found {len(image_paths)} images")
        print(f"[INFO] Output directory: {self.config.save_result}")

        for idx, image_path in enumerate(image_paths, 1):
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"[WARN] Skipping unreadable image: {image_path}")
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            stem = self.get_output_stem(str(image_path), idx)

            prob1 = self.infer_single_model(self.model1, image_rgb)
            prob2 = self.infer_single_model(self.model2, image_rgb)
            mask1 = (prob1 > self.config.bin_thresh1).astype(np.uint8) * 255
            mask2 = (prob2 > self.config.bin_thresh2).astype(np.uint8) * 255

            mask1_path = os.path.join(self.config.save_result, f"{stem}_mask1.png")
            mask2_path = os.path.join(self.config.save_result, f"{stem}_mask2.png")
            overlay_path = os.path.join(self.config.save_result, f"{stem}_overlay.png")
            mask1_overlay_path = os.path.join(self.config.save_result, f"{stem}_mask1_overlay.png")
            mask2_overlay_path = os.path.join(self.config.save_result, f"{stem}_mask2_overlay.png")
            panel_path = os.path.join(self.config.save_result, f"{stem}_analysis_panel.png")
            analysis_path = os.path.join(self.config.save_result, f"{stem}_analysis.json")

            cv2.imwrite(mask1_path, mask1)
            cv2.imwrite(mask2_path, mask2)
            create_dual_mask_overlay(
                image_path=str(image_path),
                mask1_path=mask1_path,
                mask2_path=mask2_path,
                output_path=overlay_path,
                color1=(0, 255, 0),
                color2=(0, 0, 255),
                alpha1=0.45,
                alpha2=0.45,
                blend_mode="overlay",
            )
            overlay1_rgb = create_single_mask_overlay(image_rgb, mask1, color=(0, 255, 0), alpha=0.45)
            overlay2_rgb = create_single_mask_overlay(image_rgb, mask2, color=(0, 0, 255), alpha=0.45)
            cv2.imwrite(mask1_overlay_path, cv2.cvtColor(overlay1_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(mask2_overlay_path, cv2.cvtColor(overlay2_rgb, cv2.COLOR_RGB2BGR))

            # Use original analysis visualization code path without saving copied originals in output folders.
            mask2_results = None
            with tempfile.TemporaryDirectory(prefix=f"{stem}_analysis_") as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                analyzer_mask_path = tmp_dir_path / f"{stem}_mask.png"
                cv2.imwrite(str(analyzer_mask_path), mask2)
                # Run analyzer on the temp mask so visualization has internal labeled-mask data.
                mask2_results = self.analyzer.analyze(str(analyzer_mask_path))
                if mask2_results is not None:
                    dummy_path = tmp_dir_path / "dummy.png"
                    self.analyzer.visualize(
                        str(analyzer_mask_path),
                        mask2_results,
                        str(dummy_path),
                        original_image_path=str(image_path),
                        bbox_min_area=self.config.panel_bbox_min_size,
                    )
                    generated_panel = tmp_dir_path / "visualizations" / "4_original_with_overlay.png"
                    if generated_panel.exists():
                        shutil.copyfile(str(generated_panel), panel_path)
                    else:
                        print(f"[WARN] Panel image not generated for: {image_path.name}")
                else:
                    print(f"[WARN] Mask2 analysis failed for: {image_path.name}")

            mask1_metrics = analyze_single_mask(mask1_path, self.analyzer)
            mask2_metrics = mask2_results
            mask1_bin = mask1 > 127
            mask2_bin = mask2 > 127
            overlap = np.logical_and(mask1_bin, mask2_bin)
            union = np.logical_or(mask1_bin, mask2_bin)
            total_area = mask1_bin.shape[0] * mask1_bin.shape[1]
            overlap_area = int(overlap.sum())
            union_area = int(union.sum())

            analysis_data = {
                "image": str(image_path),
                "output_files": {
                    "mask1": mask1_path,
                    "mask2": mask2_path,
                    "overlay": overlay_path,
                    "mask1_overlay": mask1_overlay_path,
                    "mask2_overlay": mask2_overlay_path,
                    "analysis_panel": panel_path,
                },
                "mask1_analysis": serialize_mask_metrics(mask1_metrics),
                "mask2_analysis": serialize_mask_metrics(mask2_metrics),
                "pairwise_analysis": {
                    "image_total_area": int(total_area),
                    "mask1_area": int(mask1_bin.sum()),
                    "mask2_area": int(mask2_bin.sum()),
                    "overlap_area": overlap_area,
                    "union_area": union_area,
                    "mask1_only_area": int(np.logical_and(mask1_bin, ~mask2_bin).sum()),
                    "mask2_only_area": int(np.logical_and(mask2_bin, ~mask1_bin).sum()),
                    "overlap_coverage_percentage": float(100.0 * overlap_area / total_area if total_area > 0 else 0.0),
                    "union_coverage_percentage": float(100.0 * union_area / total_area if total_area > 0 else 0.0),
                    "iou": float(overlap_area / union_area if union_area > 0 else 0.0),
                },
            }

            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)

            print(f"[INFO] Processed {idx}/{len(image_paths)}: {image_path.name}")


def main():
    args = parse_args()
    config = build_config(args)
    DualFeatureInference(config).run()


if __name__ == "__main__":
    main()
