#!/usr/bin/env python3
"""
Detect yellow/rust stain patterns and generate binary masks + overlays.

Example:
  python tools/detect_yellow_rust_mask.py \
    --input Irony1.jpg irony2.jpg \
    --output-dir test_result/rust_mask \
    --save-overlay
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def remove_small_components(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out


def detect_rust_mask(
    bgr: np.ndarray,
    sat_min: int = 18,
    value_min: int = 35,
    b_ab_min: int = 4,
    min_area: int = 60,
    open_k: int = 3,
    close_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    l = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    # Rust/yellow stains are typically warm (positive a,b), mildly saturated, and not near-black.
    warm_score = 0.6 * a + 0.4 * b
    warm_score = normalize_01(warm_score)
    sat_score = normalize_01(s)
    dark_gate = normalize_01(255.0 - l)

    score = 0.70 * warm_score + 0.20 * sat_score + 0.10 * dark_gate
    score_u8 = np.clip(score * 255.0, 0, 255).astype(np.uint8)

    coarse_gate = (
        (hsv[:, :, 1] >= sat_min)
        & (hsv[:, :, 2] >= value_min)
        & (lab[:, :, 2] >= (128 + b_ab_min))
    )

    otsu_thresh, _ = cv2.threshold(score_u8[coarse_gate], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) if np.any(coarse_gate) else (255, None)
    # Keep threshold stable when gate is sparse.
    thresh = int(max(70, min(210, otsu_thresh)))
    mask = np.zeros_like(score_u8, dtype=np.uint8)
    mask[(score_u8 >= thresh) & coarse_gate] = 255

    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    mask = remove_small_components(mask, min_area=min_area)
    return mask, score_u8


def make_overlay(bgr: np.ndarray, mask_u8: np.ndarray, color=(0, 255, 255), alpha: float = 0.45) -> np.ndarray:
    out = bgr.copy().astype(np.float32)
    idx = mask_u8 > 127
    out[idx] = out[idx] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def iter_input_images(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_file():
            paths.append(p)
            continue
        if p.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
                paths.extend(sorted(p.glob(ext)))
            continue
        paths.extend(sorted(Path(".").glob(item)))
    return [p for p in paths if p.is_file()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect yellow/rust stain masks from images.")
    parser.add_argument("--input", "-i", nargs="+", required=True, help="Image paths, directories, or glob patterns.")
    parser.add_argument("--output-dir", "-o", type=str, default="test_result/rust_mask", help="Output directory.")
    parser.add_argument("--save-overlay", action="store_true", help="Save color overlay image.")
    parser.add_argument("--save-score", action="store_true", help="Save grayscale warmness score map.")
    parser.add_argument("--sat-min", type=int, default=18, help="Min HSV saturation for candidate pixels.")
    parser.add_argument("--value-min", type=int, default=35, help="Min HSV value for candidate pixels.")
    parser.add_argument("--b-ab-min", type=int, default=4, help="Min Lab b offset above neutral (128).")
    parser.add_argument("--min-area", type=int, default=60, help="Remove connected components smaller than this.")
    parser.add_argument("--open-k", type=int, default=3, help="Kernel size for opening.")
    parser.add_argument("--close-k", type=int, default=5, help="Kernel size for closing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_paths = iter_input_images(args.input)
    if not in_paths:
        raise SystemExit("No input images found.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in in_paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] skip unreadable image: {p}")
            continue

        mask, score = detect_rust_mask(
            bgr=bgr,
            sat_min=args.sat_min,
            value_min=args.value_min,
            b_ab_min=args.b_ab_min,
            min_area=args.min_area,
            open_k=args.open_k,
            close_k=args.close_k,
        )

        stem = p.stem
        mask_path = out_dir / f"{stem}_rust_mask.png"
        cv2.imwrite(str(mask_path), mask)

        fg = int((mask > 127).sum())
        total = int(mask.size)
        cov = 100.0 * fg / max(1, total)
        print(f"[OK] {p.name}: mask={mask_path.name}, area={fg} px ({cov:.2f}%)")

        if args.save_overlay:
            overlay = make_overlay(bgr, mask, color=(0, 255, 255), alpha=0.45)
            overlay_path = out_dir / f"{stem}_rust_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)

        if args.save_score:
            score_path = out_dir / f"{stem}_rust_score.png"
            cv2.imwrite(str(score_path), score)


if __name__ == "__main__":
    main()
