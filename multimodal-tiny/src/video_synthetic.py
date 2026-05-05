#!/usr/bin/env python3
"""
Synthetic video dataset — generates simple moving shapes with text descriptions.
No downloads needed. Creates [B, 3, T, H, W] video clips of geometric shapes
moving across frames.
"""

import math
import random
import torch
from torch.utils.data import Dataset

# Default video parameters
VIDEO_FRAMES = 4
VIDEO_RESOLUTION = 64


def _draw_shape_on_canvas(canvas, shape_type, color, center_x, center_y, size):
    """
    Draw a geometric shape on a [3, H, W] canvas. Coordinates in [-1, 1].
    """
    H, W = canvas.shape[1], canvas.shape[2]
    # Create coordinate grid
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )

    if shape_type == 'circle':
        dist = torch.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
        mask = dist < size
    elif shape_type == 'square':
        mask = (torch.abs(x_grid - center_x) < size) & (torch.abs(y_grid - center_y) < size)
    elif shape_type == 'triangle':
        # Pointing up: y coord constraint
        half_w = size * (1 - (y_grid - center_y) / size)
        mask = (y_grid > center_y - size) & (y_grid < center_y + size) & \
               (torch.abs(x_grid - center_x) < half_w)
    elif shape_type == 'star':
        # A 4-pointed star: overlap of two squares at 45 degrees
        sq1 = (torch.abs(x_grid - center_x) < size) & (torch.abs(y_grid - center_y) < size)
        sq2 = (torch.abs(x_grid - center_y) < size) & (torch.abs(y_grid - center_x) < size)  # rotated
        # Actually let's just do a small diamond
        # Diamond: |x-cx| + |y-cy| < size
        dist_l1 = torch.abs(x_grid - center_x) + torch.abs(y_grid - center_y)
        mask = dist_l1 < size * 1.5
    else:
        mask = torch.zeros_like(x_grid, dtype=torch.bool)

    for c in range(3):
        canvas[c] = canvas[c] * (~mask).float() + color[c] * mask.float()
    return canvas


def generate_video_sample(seed=None):
    """
    Generate a synthetic video clip with a moving shape.
    Returns: (tensor[3, T, H, W], caption_string)
    """
    if seed is not None:
        random.seed(seed % 2**32)
        # Set torch seed
        torch.random.manual_seed(seed % 2**32)

    # Random shape type
    shape_types = ['circle', 'square', 'triangle', 'star']
    shape_type = random.choice(shape_types)

    # Random color (RGB in [-1, 1])
    colors = {
        'red': (1.0, -1.0, -1.0),
        'green': (-1.0, 1.0, -1.0),
        'blue': (-1.0, -1.0, 1.0),
        'yellow': (1.0, 1.0, -1.0),
        'purple': (1.0, -1.0, 1.0),
        'orange': (1.0, 0.5, -1.0),
        'cyan': (-1.0, 1.0, 1.0),
        'white': (1.0, 1.0, 1.0),
    }
    color_name = random.choice(list(colors.keys()))
    color_rgb = colors[color_name]

    # Background
    bg_colors = {
        'black': (-1.0, -1.0, -1.0),
        'dark gray': (-0.7, -0.7, -0.7),
        'navy': (-1.0, -1.0, -0.5),
        'dark green': (-1.0, -0.3, -1.0),
    }
    bg_name = random.choice(list(bg_colors.keys()))
    bg_rgb = bg_colors[bg_name]

    # Object size (0.1-0.25 in normalized coords)
    size = random.uniform(0.1, 0.2)

    # Movement direction and speed
    directions = [
        ('left to right', 1.0, 0.0),
        ('right to left', -1.0, 0.0),
        ('top to bottom', 0.0, 1.0),
        ('bottom to top', 0.0, -1.0),
        ('diagonal', 0.8, 0.6),
        ('diagonal', -0.7, 0.7),
    ]
    dir_name, dx, dy = random.choice(directions)
    speed = random.uniform(0.15, 0.3)

    # Generate frames
    T = VIDEO_FRAMES
    H = W = VIDEO_RESOLUTION
    video = torch.zeros(3, T, H, W)

    for t in range(T):
        # Position moves linearly across frames
        progress = -1.0 + 2.0 * t / (T - 1) if T > 1 else 0.0  # -1 to 1
        cx = progress * speed * dx
        cy = progress * speed * dy

        # Add some randomness to starting position
        offset_x = random.uniform(-0.3, 0.3)
        offset_y = random.uniform(-0.3, 0.3)
        cx += offset_x
        cy += offset_y

        # Clamp to [-1, 1]
        cx = max(-0.8, min(0.8, cx))
        cy = max(-0.8, min(0.8, cy))

        # Initialize frame with background
        frame = torch.zeros(3, H, W)
        for c in range(3):
            frame[c] = bg_rgb[c]

        # Draw shape
        frame = _draw_shape_on_canvas(frame, shape_type, color_rgb, cx, cy, size)
        video[:, t] = frame

    # Generate caption
    templates = [
        f"A {color_name} {shape_type} moving {dir_name} on a {bg_name} background.",
        f"A {color_name} {shape_type} that moves {dir_name} on a {bg_name} background.",
        f"{color_name.title()} {shape_type} moving {dir_name} across {bg_name} background.",
    ]
    # Simpler template for the model
    caption = f"A {color_name} {shape_type} moving {dir_name} on a {bg_name} background."

    return video, caption


class VideoDataset(Dataset):
    """Synthetic video dataset: (video_clip, caption) pairs.

    Video shape: [3, T, H, W] where T=4, H=W=64.
    Normalized to range [-1, 1].
    """

    def __init__(self, num_samples=3000, video_frames=VIDEO_FRAMES,
                 video_resolution=VIDEO_RESOLUTION, seed=None):
        self.num_samples = num_samples
        self.video_frames = video_frames
        self.video_resolution = video_resolution
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        video, caption = generate_video_sample(seed=(self.seed or 0) + idx)
        return video, caption


if __name__ == "__main__":
    # Test
    video, caption = generate_video_sample(seed=42)
    print(f"Video shape: {video.shape}")  # [3, 4, 64, 64]
    print(f"Caption: {caption}")
    print(f"Range: [{video.min():.3f}, {video.max():.3f}]")

    ds = VideoDataset(num_samples=3)
    for i in range(3):
        v, c = ds[i]
        print(f"[{i}] {v.shape} - {c}")
    print("Video synthetic data: OK ✓")
