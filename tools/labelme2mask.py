# -*- coding: utf-8 -*-
"""
Labelme 实例标注 -> P 模式 PNG 掩码
- 输入目录: INPUT_DIR 中的 *.json (labelme 标注)
- 输出目录: OUTPUT_DIR 中的 *.png (P 模式，索引掩码) 与 *_instances.tsv (索引到实例映射)
- 一个实例一个颜色；调色盘 256 色（索引 0 = 背景）
- 识别 shape_type: polygon / rectangle / circle；其余类型跳过
"""

import os
import json
import math
import colorsys
from PIL import Image, ImageDraw

# ==================== 可按需修改 ====================
INPUT_DIR = r"D:\Dataset\DJI\labelme_all_data"
OUTPUT_DIR = r"D:\Dataset\DJI\labelme_all_data\p_mode_mask"
BACKGROUND_INDEX = 0  # 背景索引
MAX_COLORS = 256  # P 模式色数上限（固定 256）
SAVE_INDEX_MAP = False  # 是否输出 *_instances.tsv


# ====================================================

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_palette_256():
    """
    生成 256*3 的调色盘列表。索引 0=黑色，其余使用近似均匀的 HSV 取样。
    注意：掩码训练通常看重索引值，本调色盘仅用于可视化。
    """
    palette = [0] * (MAX_COLORS * 3)
    # idx 0: (0,0,0)
    palette[0:3] = [0, 0, 0]
    golden = 0.618033988749895  # 黄金比间隔，避免相邻色过近
    for i in range(1, MAX_COLORS):
        h = (i * golden) % 1.0
        s, v = 0.8, 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        palette[i * 3 + 0] = int(r * 255)
        palette[i * 3 + 1] = int(g * 255)
        palette[i * 3 + 2] = int(b * 255)
    return palette


def rect_to_poly(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def round_point(pt):
    return int(round(pt[0])), int(round(pt[1]))


def draw_shape(draw: ImageDraw.ImageDraw, shape: dict, fill_index: int):
    st = (shape.get("shape_type") or "polygon").lower()
    pts = shape.get("points") or []

    if st == "polygon" and len(pts) >= 3:
        poly = [round_point(p) for p in pts]
        draw.polygon(poly, fill=fill_index)

    elif st == "rectangle" and len(pts) >= 2:
        poly = [round_point(p) for p in rect_to_poly(pts[0], pts[1])]
        draw.polygon(poly, fill=fill_index)

    elif st == "circle" and len(pts) >= 2:
        # labelme 的 circle: [center, edge_point]
        (cx, cy) = pts[0]
        (px, py) = pts[1]
        r = math.hypot(px - cx, py - cy)
        bbox = [int(round(cx - r)), int(round(cy - r)),
                int(round(cx + r)), int(round(cy + r))]
        draw.ellipse(bbox, fill=fill_index)

    else:
        print(f"[WARN] Unsupported shape_type '{st}', skip.")


def instance_key(shape: dict):
    """
    用 (label, group_id) 表示同一实例。若无 group_id，则每个 shape 单独成实例。
    """
    label = shape.get("label", "")
    gid = shape.get("group_id")
    if gid is None:
        # 对没有 group_id 的 shape，使用其对象 id 作为唯一键（仅在当前图内唯一即可）
        return "__auto__", id(shape)
    return label, gid


def process_one_json(json_path: str, out_dir: str, palette: list):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h = data.get("imageHeight")
    w = data.get("imageWidth")
    if not (isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0):
        raise ValueError(f"{json_path}: 缺少有效的 imageHeight/imageWidth。")

    img = Image.new("P", (w, h), BACKGROUND_INDEX)
    img.putpalette(palette)
    draw = ImageDraw.Draw(img)

    inst2index = {}  # 实例键 -> 调色盘索引
    next_idx = 1  # 从 1 开始分配（0 留给背景）

    shapes = data.get("shapes") or []
    for shape in shapes:
        key = instance_key(shape)
        if key not in inst2index:
            if next_idx >= MAX_COLORS:
                # 超过 255 个实例：将多余实例映射到最后一个索引（255）
                print("[WARN] 实例数超过 255，后续实例将复用索引 255。")
                inst2index[key] = MAX_COLORS - 1
            else:
                inst2index[key] = next_idx
                next_idx += 1

        draw_shape(draw, shape, inst2index[key])

    base = os.path.splitext(os.path.basename(json_path))[0]
    out_png = os.path.join(out_dir, f"{base}.png")
    img.save(out_png, format="PNG", optimize=True)
    print(f"[OK] {out_png}")

    if SAVE_INDEX_MAP:
        # 生成索引映射（index\tlabel\tgroup_id）
        mapping_path = os.path.join(out_dir, f"{base}_instances.tsv")
        # 反向：index -> [(label, group_id)]
        idx2items = {}
        for k, v in inst2index.items():
            idx2items.setdefault(v, []).append(k)

        with open(mapping_path, "w", encoding="utf-8") as mf:
            mf.write("index\tlabel\tgroup_id\n")
            for idx in sorted(idx2items.keys()):
                for (label, gid) in idx2items[idx]:
                    if label == "__auto__":
                        # 无 group_id 的自动实例
                        mf.write(f"{idx}\t\t\n")
                    else:
                        mf.write(f"{idx}\t{label}\t{gid}\n")
        print(f"[OK] {mapping_path}")


def main():
    ensure_dirs()
    palette = generate_palette_256()

    json_files = [os.path.join(INPUT_DIR, f)
                  for f in os.listdir(INPUT_DIR)
                  if f.lower().endswith(".json")]

    if not json_files:
        print(f"[INFO] {INPUT_DIR} 下未发现 *.json 标注文件。")
        return

    for jp in sorted(json_files):
        try:
            process_one_json(jp, OUTPUT_DIR, palette)
        except Exception as e:
            print(f"[ERR] 处理失败: {jp} -> {e}")


if __name__ == "__main__":
    main()
