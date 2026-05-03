#!/usr/bin/env python3
"""
Chinese Synthetic Data — wraps existing generators with Chinese captions.
Pixels/frames/specs stay the same; only text captions become Chinese.
"""

import random
import torch
from torch.utils.data import Dataset
from src.synthetic_data import SyntheticDataset as EnImageDataset
from src.audio_synthetic import AudioDataset as EnAudioDataset
from src.video_synthetic import VideoDataset as EnVideoDataset

# ── Chinese Caption Templates ──────────────────────────────────────

SHAPE_ZH = {
    'circle': '圆形', 'square': '正方形', 'triangle': '三角形',
    'star': '星形', 'heart': '心形', 'diamond': '菱形',
}

COLOR_ZH = {
    'red': '红色', 'green': '绿色', 'blue': '蓝色',
    'yellow': '黄色', 'purple': '紫色', 'orange': '橙色',
    'cyan': '青色', 'pink': '粉色', 'white': '白色',
    'black': '黑色', 'gray': '灰色', 'navy': '深蓝',
    'dark gray': '深灰', 'dark green': '深绿',
}

BG_ZH = {
    'black': '黑色', 'dark gray': '深灰色', 'navy': '深蓝色',
    'dark green': '深绿色',
}

DIR_ZH = {
    'left to right': '从左到右',
    'right to left': '从右到左',
    'top to bottom': '从上到下',
    'bottom to top': '从下到上',
    'diagonal': '对角线',
}

SND_ZH = {
    'pure tone': '纯音',
    'high': '高频',
    'low': '低频',
    'sine': '正弦波',
    'noise': '噪音',
    'background static': '背景噪音',
    'soft': '柔和',
    'harsh': '刺耳',
    'pure tone high': '高频纯音',
    'pure tone medium': '中频纯音',
    'pure tone low': '低频纯音',
    'background noise': '背景噪音',
    'white noise': '白噪音',
}


def image_caption_zh(caption_en):
    """Translate English image caption to Chinese."""
    # Parse English template: "A [color] [shape] on a [bg] background."
    # or "An image containing [color] [shape] and [color] [shape] on [bg] background."
    
    words = caption_en.lower().replace('.', '').split()
    
    # Find colors and shapes
    found_colors = [w for w in words if w in COLOR_ZH]
    found_shapes = [w for w in words if w in SHAPE_ZH]
    found_bg = next((w for w in words if w in BG_ZH), 'black')
    
    if not found_colors or not found_shapes:
        return caption_en  # fallback
    
    if len(found_colors) >= 2 and len(found_shapes) >= 2:
        # Multi-object: "A [c1] [s1] and a [c2] [s2] on [bg] background."
        t1 = f"有一个{COLOR_ZH[found_colors[0]]}{SHAPE_ZH[found_shapes[0]]}和{COLOR_ZH[found_colors[1]]}{SHAPE_ZH[found_shapes[1]]}在{BG_ZH[found_bg]}背景上"
    else:
        c = COLOR_ZH.get(found_colors[0], '红色')
        s = SHAPE_ZH.get(found_shapes[0], '圆形')
        bg = BG_ZH.get(found_bg, '黑色')
        if 'small' in words:
            t1 = f"一个小的{c}{s}在{bg}背景上"
        else:
            t1 = f"一个{c}{s}在{bg}背景上"
    
    return t1


def audio_caption_zh(caption_en):
    """Translate English audio caption to Chinese."""
    caption_lower = caption_en.lower()
    for eng, zh in sorted(SND_ZH.items(), key=lambda x: -len(x[0])):
        if eng in caption_lower:
            return zh
    return caption_en


def video_caption_zh(caption_en):
    """Translate English video caption to Chinese."""
    # Template: "A [color] [shape] moving [direction] on a [bg] background."
    words = caption_en.lower().replace('.', '').split()
    
    found_colors = [w for w in words if w in COLOR_ZH]
    found_shapes = [w for w in words if w in SHAPE_ZH]
    found_bg = next((w for w in words if w in BG_ZH), 'black')
    found_dir = next((d for d in DIR_ZH if d in caption_en.lower()), 'left to right')
    
    c = COLOR_ZH.get(found_colors[0], '红色') if found_colors else '红色'
    s = SHAPE_ZH.get(found_shapes[0], '圆形') if found_shapes else '圆形'
    bg = BG_ZH.get(found_bg, '黑色')
    d = DIR_ZH.get(found_dir, '从左到右')
    
    return f"一个{c}{s}在{bg}背景上{d}移动"


# ── Chinese Datasets ────────────────────────────────────────────────

class ZhImageDataset(Dataset):
    """Image dataset with Chinese captions."""
    def __init__(self, num_samples=10000, image_size=224, seed=None):
        self._ds = EnImageDataset(num_samples=num_samples, image_size=image_size, seed=seed)
    
    def __len__(self):
        return len(self._ds)
    
    def __getitem__(self, idx):
        img, cap_en = self._ds[idx]
        return img, image_caption_zh(cap_en)


class ZhAudioDataset(Dataset):
    """Audio dataset with Chinese captions."""
    def __init__(self, num_samples=5000, seed=None):
        self._ds = EnAudioDataset(num_samples=num_samples, seed=seed)
    
    def __len__(self):
        return len(self._ds)
    
    def __getitem__(self, idx):
        mel, cap_en = self._ds[idx]
        return mel, audio_caption_zh(cap_en)


class ZhVideoDataset(Dataset):
    """Video dataset with Chinese captions."""
    def __init__(self, num_samples=5000, seed=None):
        self._ds = EnVideoDataset(num_samples=num_samples, seed=seed)
    
    def __len__(self):
        return len(self._ds)
    
    def __getitem__(self, idx):
        vid, cap_en = self._ds[idx]
        return vid, video_caption_zh(cap_en)


# ── Test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Testing Chinese Captions ===")
    
    # Test image
    ds_img = ZhImageDataset(num_samples=5)
    for i in range(5):
        img, cap = ds_img[i]
        print(f"[Img {i}] {cap[:60]}")
    
    print()
    ds_aud = ZhAudioDataset(num_samples=5)
    for i in range(5):
        mel, cap = ds_aud[i]
        print(f"[Aud {i}] {cap[:60]}")
    
    print()
    ds_vid = ZhVideoDataset(num_samples=5)
    for i in range(5):
        vid, cap = ds_vid[i]
        print(f"[Vid {i}] {cap[:60]}")
    
    print("\nChinese data generators: OK ✓")
