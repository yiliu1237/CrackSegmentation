from __future__ import division

import argparse
import os
import sys
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

from tools.defect_mask_overlay import create_dual_mask_overlay
from tools.detect_yellow_rust_mask import detect_rust_mask, make_overlay

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


@dataclass
class CrackRustConfig:
    task: str = "crack_rust_pipeline"
    images_dir: str = ""
    checkpoint1: str = ""
    save_result: str = ""
    save_result_root: str = "./test_result"
    dinov3_arch: str = "convnext_base"
    dinov3_ckpt: str = "./pretrained_checkpoint/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"
    bin_thresh1: float = 0.5
    patch_size: int = 512
    patch_stride_ratio: float = 0.66
    infer_batch_size: int = 16
    output_name_mode: str = "original"  # "original" or "index"
    output_prefix: str = ""
    # Rust detector params
    rust_sat_min: int = 18
    rust_value_min: int = 35
    rust_b_ab_min: int = 4
    rust_min_area: int = 60
    rust_open_k: int = 3
    rust_close_k: int = 5
    save_rust_score: bool = True


def parse_args():
    parser = argparse.ArgumentParser(description="Crack + rust dual-feature inference pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument("--image-dir", type=str, default=None, help="Input image directory.")
    parser.add_argument("--checkpoint1", type=str, default=None, help="Checkpoint path for crack model.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--bin-thresh1", type=float, default=None, help="Binary threshold for crack mask.")
    parser.add_argument("--output-prefix", type=str, default=None, help="Prefix for output filenames.")
    parser.add_argument(
        "--output-name-mode",
        type=str,
        choices=["original", "index"],
        default=None,
        help="Naming mode: original image stem or running index.",
    )
    parser.add_argument("--rust-sat-min", type=int, default=None, help="Rust detector min HSV saturation.")
    parser.add_argument("--rust-value-min", type=int, default=None, help="Rust detector min HSV value.")
    parser.add_argument("--rust-b-ab-min", type=int, default=None, help="Rust detector min Lab b* offset.")
    parser.add_argument("--rust-min-area", type=int, default=None, help="Rust detector min connected-component area.")
    parser.add_argument("--rust-open-k", type=int, default=None, help="Rust detector morphology open kernel.")
    parser.add_argument("--rust-close-k", type=int, default=None, help="Rust detector morphology close kernel.")
    parser.add_argument("--save-rust-score", dest="save_rust_score", action="store_true", help="Save rust score map.")
    parser.add_argument("--no-save-rust-score", dest="save_rust_score", action="store_false", help="Do not save rust score map.")
    parser.set_defaults(save_rust_score=None)
    return parser.parse_args()


def load_config(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("crack_rust_inference", data.get("inference", data))


def build_config(args):
    cfg = CrackRustConfig()
    merged = asdict(cfg)
    merged.update(load_config(args.config))

    if args.image_dir is not None:
        merged["images_dir"] = args.image_dir
    if args.checkpoint1 is not None:
        merged["checkpoint1"] = args.checkpoint1
    if args.output_dir is not None:
        merged["save_result"] = args.output_dir
    if args.bin_thresh1 is not None:
        merged["bin_thresh1"] = args.bin_thresh1
    if args.output_prefix is not None:
        merged["output_prefix"] = args.output_prefix
    if args.output_name_mode is not None:
        merged["output_name_mode"] = args.output_name_mode
    if args.rust_sat_min is not None:
        merged["rust_sat_min"] = args.rust_sat_min
    if args.rust_value_min is not None:
        merged["rust_value_min"] = args.rust_value_min
    if args.rust_b_ab_min is not None:
        merged["rust_b_ab_min"] = args.rust_b_ab_min
    if args.rust_min_area is not None:
        merged["rust_min_area"] = args.rust_min_area
    if args.rust_open_k is not None:
        merged["rust_open_k"] = args.rust_open_k
    if args.rust_close_k is not None:
        merged["rust_close_k"] = args.rust_close_k
    if args.save_rust_score is not None:
        merged["save_rust_score"] = args.save_rust_score

    if not merged.get("images_dir"):
        raise ValueError("images_dir is required. Set config or pass --image-dir.")
    if not merged.get("checkpoint1"):
        raise ValueError("checkpoint1 is required. Set config or pass --checkpoint1.")
    if not merged.get("save_result"):
        merged["save_result"] = os.path.join(merged["save_result_root"], merged["task"])

    return CrackRustConfig(**merged)


def list_images(images_dir):
    exts = ("*.JPG", "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for ext in exts:
        files.extend(Path(images_dir).glob(ext))
    return sorted(set(files))


def create_single_mask_overlay(image_rgb, mask_u8, color=(0, 255, 0), alpha=0.45):
    overlay = image_rgb.copy().astype(np.float32)
    mask_bin = (mask_u8 > 127).astype(np.float32)
    for c in range(3):
        overlay[:, :, c] = (
            image_rgb[:, :, c].astype(np.float32) * (1.0 - alpha * mask_bin) +
            float(color[c]) * alpha * mask_bin
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_combined_mask(crack_mask_u8, rust_mask_u8):
    crack_bin = crack_mask_u8 > 127
    rust_bin = rust_mask_u8 > 127
    combined = np.zeros_like(crack_mask_u8, dtype=np.uint8)
    combined[np.logical_and(crack_bin, ~rust_bin)] = 85
    combined[np.logical_and(~crack_bin, rust_bin)] = 170
    combined[np.logical_and(crack_bin, rust_bin)] = 255
    return combined


class CrackRustInference:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_to_tensor = ImgToTensor()
        self.model1 = self.init_model()

    def init_model(self):
        model = CrackModelDinoV3(
            dinov3_arch=self.config.dinov3_arch,
            dinov3_ckpt=self.config.dinov3_ckpt,
        ).to(self.device)
        state_dict = torch.load(self.config.checkpoint1, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def infer_crack_prob(self, image_rgb):
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
                model_results.extend(torch.sigmoid(self.model1(each)).cpu())
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

            crack_prob = self.infer_crack_prob(image_rgb)
            crack_mask = (crack_prob > self.config.bin_thresh1).astype(np.uint8) * 255
            rust_mask, rust_score = detect_rust_mask(
                bgr=image_bgr,
                sat_min=self.config.rust_sat_min,
                value_min=self.config.rust_value_min,
                b_ab_min=self.config.rust_b_ab_min,
                min_area=self.config.rust_min_area,
                open_k=self.config.rust_open_k,
                close_k=self.config.rust_close_k,
            )

            crack_mask_path = os.path.join(self.config.save_result, f"{stem}_crack_mask.png")
            crack_overlay_path = os.path.join(self.config.save_result, f"{stem}_crack_overlay.png")
            rust_mask_path = os.path.join(self.config.save_result, f"{stem}_rust_mask.png")
            rust_overlay_path = os.path.join(self.config.save_result, f"{stem}_rust_overlay.png")
            rust_score_path = os.path.join(self.config.save_result, f"{stem}_rust_score.png")
            dual_overlay_path = os.path.join(self.config.save_result, f"{stem}_dual_overlay.png")
            combined_mask_path = os.path.join(self.config.save_result, f"{stem}_combined_mask.png")

            cv2.imwrite(crack_mask_path, crack_mask)
            cv2.imwrite(rust_mask_path, rust_mask)

            crack_overlay_rgb = create_single_mask_overlay(image_rgb, crack_mask, color=(0, 255, 0), alpha=0.45)
            rust_overlay = make_overlay(image_bgr, rust_mask, color=(0, 255, 255), alpha=0.45)
            cv2.imwrite(crack_overlay_path, cv2.cvtColor(crack_overlay_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(rust_overlay_path, rust_overlay)
            if self.config.save_rust_score:
                cv2.imwrite(rust_score_path, rust_score)

            create_dual_mask_overlay(
                image_path=str(image_path),
                mask1_path=crack_mask_path,
                mask2_path=rust_mask_path,
                output_path=dual_overlay_path,
                color1=(0, 255, 0),
                color2=(255, 255, 0),
                alpha1=0.45,
                alpha2=0.45,
                blend_mode="overlay",
            )

            combined_mask = create_combined_mask(crack_mask, rust_mask)
            cv2.imwrite(combined_mask_path, combined_mask)

            print(f"[INFO] Processed {idx}/{len(image_paths)}: {image_path.name}")


def main():
    args = parse_args()
    config = build_config(args)
    CrackRustInference(config).run()


if __name__ == "__main__":
    main()
