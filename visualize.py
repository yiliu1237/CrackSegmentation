from __future__ import division

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import yaml

from dataloaders import ImgToTensor
from model import CrackModelDinoV3
import utils

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


@dataclass
class InferenceConfig:
    task: str = "default_task"
    images_dir: str = ""
    checkpoint: str = ""
    save_result: str = ""
    save_result_root: str = "./test_result"
    dinov3_ckpt: str = "./pretrained_checkpoint/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"
    overlay: bool = True
    save_mask_01: bool = True
    save_heatmap: bool = True
    save_comparison: bool = False
    bin_thresh: float = 0.5
    patch_size: int = 512
    patch_stride_ratio: float = 0.66
    infer_batch_size: int = 16
    output_name_mode: str = "original"  # "original" or "index"
    output_prefix: str = ""
    heatmap_colormap: str = "turbo"  # "turbo" or "jet"


def parse_args():
    parser = argparse.ArgumentParser(description="Run crack segmentation inference and visualization.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--image-dir", type=str, default=None, help="Input image directory.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save outputs.")
    parser.add_argument("--output-prefix", type=str, default=None, help="Prefix for output filenames.")
    parser.add_argument(
        "--output-name-mode",
        type=str,
        choices=["original", "index"],
        default=None,
        help="Naming mode: original image stem or running index.",
    )
    parser.add_argument("--overlay", dest="overlay", action="store_true", help="Enable overlay output.")
    parser.add_argument("--no-overlay", dest="overlay", action="store_false", help="Disable overlay output.")
    parser.set_defaults(overlay=None)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("inference", data)


def build_config(args):
    cfg = InferenceConfig()
    merged = asdict(cfg)
    merged.update(load_config(args.config))
    if args.image_dir is not None:
        merged["images_dir"] = args.image_dir
    if args.checkpoint is not None:
        merged["checkpoint"] = args.checkpoint
    if args.output_dir is not None:
        merged["save_result"] = args.output_dir
    if args.output_prefix is not None:
        merged["output_prefix"] = args.output_prefix
    if args.output_name_mode is not None:
        merged["output_name_mode"] = args.output_name_mode
    if args.overlay is not None:
        merged["overlay"] = args.overlay
    if not merged.get("images_dir"):
        raise ValueError("images_dir is required. Set it in config ('inference.images_dir') or pass --image-dir.")
    if not merged.get("checkpoint"):
        raise ValueError("checkpoint is required. Set it in config ('inference.checkpoint') or pass --checkpoint.")
    if not merged.get("save_result"):
        merged["save_result"] = os.path.join(merged["save_result_root"], merged["task"])
    return InferenceConfig(**merged)


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_model()
        self.test()

    def init_model(self):
        self.log('Initializing model')
        self.model = CrackModelDinoV3(
            dinov3_arch='convnext_base',
            dinov3_ckpt=self.config.dinov3_ckpt
        ).to(self.device)

        self.log(f"Restoring from checkpoint: {self.config.checkpoint}")
        self.log(self.model.load_state_dict(torch.load(self.config.checkpoint, map_location="cpu"), strict=True))

    def test(self):
        self.model.eval()
        os.makedirs(self.config.save_result, exist_ok=True)
        self.log(f"Saving outputs to: {self.config.save_result}")

        img_totensor = ImgToTensor()
        img_fnames = [path.name for path in Path(self.config.images_dir).glob("*.JPG")]
        img_fnames.extend([path.name for path in Path(self.config.images_dir).glob("*.jpg")])
        img_fnames.extend([path.name for path in Path(self.config.images_dir).glob("*.jpeg")])
        img_fnames.extend([path.name for path in Path(self.config.images_dir).glob("*.png")])
        img_fnames = sorted(set(img_fnames))
        self.log(f"Found {len(img_fnames)} images in: {self.config.images_dir}")
        if len(img_fnames) == 0:
            raise ValueError(
                f"No images found in '{self.config.images_dir}'. "
                "Supported extensions: .JPG, .jpg, .jpeg, .png"
            )
        
        for i, fname in enumerate(img_fnames):
            fpath = os.path.join(self.config.images_dir, fname)
            print(fpath)
            image = cv2.imread(fpath)
            if image is None:
                self.log(f"Skipping unreadable image: {fpath}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape
            original_h, original_w = h, w

            with torch.no_grad():
                image = img_totensor(image).squeeze(0)
                patches, patch_locs = utils.get_img_patches_adjusted(
                    image, self.config.patch_size, stride_ratio=self.config.patch_stride_ratio
                )
                patches = patches.to(self.device, non_blocking=True)
                patch_list = torch.split(patches, self.config.infer_batch_size)
                model_results = []
                for each in patch_list:
                    model_results.extend(torch.sigmoid(self.model(each)).cpu())
                pred_mask = torch.stack(model_results, 0)
                full_mask = utils.merge_pred_patches_adjusted(image, pred_mask, patch_locs)

                self.save_sample(fpath, full_mask.detach().numpy(), index=i + 1,
                                 original_size=(original_w, original_h))

    def get_output_stem(self, img_path, index):
        if self.config.output_name_mode == "index":
            base = str(index)
        else:
            base = os.path.splitext(os.path.basename(img_path))[0]
        if self.config.output_prefix:
            return f"{self.config.output_prefix}{base}"
        return base

    def save_sample(self, img_path, msk_pred, index=1, original_size=None):
        img = cv2.imread(img_path)
        if img is None:
            self.log(f"Skipping save; failed to read image: {img_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        os.makedirs(self.config.save_result, exist_ok=True)
        output_stem = self.get_output_stem(img_path, index)
        msk_pred = np.squeeze(msk_pred)

        if original_size is not None:
            msk_pred = cv2.resize(msk_pred, original_size, interpolation=cv2.INTER_NEAREST)
        else:
            msk_pred = msk_pred

        prob_map = np.clip(msk_pred, 0.0, 1.0)
        bin_mask = (msk_pred > self.config.bin_thresh).astype(np.uint8)

        if self.config.save_mask_01:
            out01 = os.path.join(self.config.save_result, f"{output_stem}_mask.png")
            cv2.imwrite(out01, bin_mask * 255)  # scale to 0-255 for visibility

        if self.config.save_heatmap:
            cmap = cv2.COLORMAP_TURBO if str(self.config.heatmap_colormap).lower() == "turbo" else cv2.COLORMAP_JET
            heatmap_u8 = (prob_map * 255.0).astype(np.uint8)
            heatmap_bgr = cv2.applyColorMap(heatmap_u8, cmap)
            out_heatmap = os.path.join(self.config.save_result, f"{output_stem}_prob_heatmap.jpg")
            cv2.imwrite(out_heatmap, heatmap_bgr)

        if self.config.overlay:
            overlay = img.copy()
            overlay[msk_pred > self.config.bin_thresh] = [0, 0, 255]
            alpha = 0.5
            overlayed_img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

            out_overlay = os.path.join(self.config.save_result, f"{output_stem}_overlay.jpg")
            cv2.imwrite(out_overlay, cv2.cvtColor(overlayed_img, cv2.COLOR_RGB2BGR))
            
        if self.config.save_comparison:
            _, axs = plt.subplots(1, 2, figsize=(30, 15))
            plt.tight_layout()
            axs = axs.ravel()

            axs[0].axis('off')
            axs[0].imshow(img / 255.)

            axs[1].axis('off')
            axs[1].imshow(msk_pred * 255, cmap='gray')

            out_cmp = os.path.join(self.config.save_result, f"{output_stem}_compare.png")
            plt.savefig(out_cmp)
            plt.close()

    def log(self, msg):
        print(f'[] {msg}')


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)
    Trainer(config)
