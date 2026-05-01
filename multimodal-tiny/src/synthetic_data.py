#!/usr/bin/env python3
"""
Synthetic multimodal dataset — no downloads needed.
Generates image-caption pairs programmatically.
"""

import math
import random
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw


# ── Synthetic data templates ───────────────────────────────────────────

SHAPES = ["circle", "square", "triangle", "star", "diamond", "heart", "cross"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "white"]
POSITIONS = ["center", "top-left", "top-right", "bottom-left", "bottom-right", "left", "right", "top", "bottom"]
SIZES = ["small", "medium", "large"]
BACKGROUNDS = ["black", "dark blue", "dark gray", "navy", "dark green"]


def _render_shape(draw, shape, color, bbox):
    """Draw a shape on the image."""
    fill_map = {
        "red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0),
        "yellow": (255, 255, 0), "purple": (128, 0, 128), "orange": (255, 165, 0),
        "pink": (255, 192, 203), "white": (255, 255, 255),
    }
    fill = fill_map.get(color, (255, 255, 255))
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    r = (x1 - x0) // 2

    if shape == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=fill)
    elif shape == "square":
        draw.rectangle([x0, y0, x1, y1], fill=fill)
    elif shape == "triangle":
        draw.polygon([(cx, y0), (x0, y1), (x1, y1)], fill=fill)
    elif shape == "star":
        points = []
        for i in range(5):
            angle = math.pi / 2 + 2 * math.pi * i / 5
            points.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
            angle += 2 * math.pi / 10
            points.append((cx + r/2 * math.cos(angle), cy - r/2 * math.sin(angle)))
        draw.polygon(points, fill=fill)
    elif shape == "diamond":
        draw.polygon([(cx, y0), (x1, cy), (cx, y1), (x0, cy)], fill=fill)
    elif shape == "heart":
        draw.ellipse([x0, y0, cx, y1], fill=fill)
        draw.ellipse([cx, y0, x1, y1], fill=fill)
        draw.polygon([(x0, y1), (cx, y1 + (y1-y0)//2), (x1, y1)], fill=fill)
    elif shape == "cross":
        w, h = (x1 - x0) // 4, (y1 - y0) // 4
        draw.rectangle([cx-w, y0, cx+w, y1], fill=fill)
        draw.rectangle([x0, cy-h, x1, cy+h], fill=fill)


def generate_sample(image_size=224, seed=None):
    """Generate a synthetic (image, caption) pair."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    bg = rng.choice(BACKGROUNDS)
    bg_map = {"black": (0, 0, 0), "dark blue": (0, 0, 50), "dark gray": (30, 30, 30),
              "navy": (0, 0, 80), "dark green": (0, 30, 0)}
    bg_color = bg_map[bg]

    img = Image.new("RGB", (image_size, image_size), bg_color)
    draw = ImageDraw.Draw(img)

    num_shapes = rng.randint(1, 3)
    shape_info = []

    for _ in range(num_shapes):
        shape = rng.choice(SHAPES)
        color = rng.choice(COLORS)
        size = rng.choice(SIZES)

        size_map = {"small": 0.15, "medium": 0.25, "large": 0.35}
        s = size_map[size]
        margin = image_size * 0.1
        x0 = rng.randint(int(margin), int(image_size * 0.6))
        y0 = rng.randint(int(margin), int(image_size * 0.6))
        x1 = x0 + int(image_size * s)
        y1 = y0 + int(image_size * s)

        _render_shape(draw, shape, color, (x0, y0, x1, y1))
        shape_info.append((shape, color, size))

    # Generate caption
    if num_shapes == 1:
        s, c, sz = shape_info[0]
        caption = f"A {sz} {c} {s} on a {bg} background."
    elif num_shapes == 2:
        parts = [f"A {c} {s}" for s, c, _ in shape_info]
        caption = f"{parts[0]} and {parts[1].lower().lower()} on a {bg} background."
    else:
        parts = [f"a {c} {s}" for s, c, _ in shape_info]
        caption = f"An image containing {', '.join(parts[:-1])}, and {parts[-1]} on a {bg} background."

    return img, caption


class SyntheticDataset(Dataset):
    """Generates synthetic image-caption pairs on the fly."""

    def __init__(self, num_samples=50000, image_size=224, seed=42):
        self.num_samples = num_samples
        self.image_size = image_size
        self.rng = random.Random(seed)
        print(f"  SyntheticDataset: {num_samples} samples, {image_size}x{image_size}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, caption = generate_sample(self.image_size, seed=idx + 42)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        return img_tensor, caption


def generate_preview(num=5, image_size=224):
    """Generate preview images to inspect."""
    from PIL import Image
    results = []
    for i in range(num):
        img, caption = generate_sample(image_size, seed=i)
        results.append((img, caption))
        print(f"  [{i}] {caption}")
    return results


if __name__ == "__main__":
    print("Synthetic data preview:")
    generate_preview(5)
    print()

    ds = SyntheticDataset(1000)
    img, cap = ds[0]
    print(f"  Sample: img={img.shape}, caption={cap[:60]}")
    print("  Synthetic data pipeline: OK ✓")
