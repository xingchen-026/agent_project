#!/usr/bin/env python3
"""
Chinese Synthetic Data — native Chinese caption templates.
Pixels/frames/specs use the same generators as English; captions are
generated directly in Chinese (not translated).
"""

import math
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

from data.synthetic import (
    _render_shape, SHAPES, COLORS, POSITIONS, SIZES, BACKGROUNDS,
)
from data.audio_synthetic import (
    _sine_wave, _square_wave, _white_noise,
    _amplitude_modulated, _frequency_modulated,
    waveform_to_mel, N_FRAMES, SAMPLE_RATE, DURATION,
)
from data.video_synthetic import (
    _draw_shape_on_canvas, VIDEO_FRAMES, VIDEO_RESOLUTION,
)

# ── Chinese vocabulary mappings ─────────────────────────────────────

SHAPE_ZH = {
    'circle': '圆形', 'square': '正方形', 'triangle': '三角形',
    'star': '星形', 'heart': '心形', 'diamond': '菱形', 'cross': '十字形',
}

COLOR_ZH = {
    'red': '红色', 'green': '绿色', 'blue': '蓝色',
    'yellow': '黄色', 'purple': '紫色', 'orange': '橙色',
    'cyan': '青色', 'pink': '粉色', 'white': '白色',
    'black': '黑色', 'gray': '灰色', 'navy': '深蓝',
    'dark gray': '深灰', 'dark green': '深绿',
}

BG_ZH = {
    'black': '黑色', 'dark blue': '深蓝色', 'dark gray': '深灰色',
    'navy': '深蓝色', 'dark green': '深绿色',
}

SIZE_ZH = {'small': '小', 'medium': '中等', 'large': '大'}

POS_ZH = {
    'center': '中央', 'top-left': '左上角', 'top-right': '右上角',
    'bottom-left': '左下角', 'bottom-right': '右下角',
    'left': '左侧', 'right': '右侧', 'top': '上方', 'bottom': '下方',
}

DIR_ZH = {
    'left to right': '从左到右',
    'right to left': '从右到左',
    'top to bottom': '从上到下',
    'bottom to top': '从下到上',
    'diagonal': '对角线',
}


# ── Image: Chinese-native caption generation ────────────────────────

def _generate_image_cn(image_size=224, seed=None):
    """Generate (PIL Image, Chinese caption) with native templates."""
    rng = random.Random(seed)

    bg = rng.choice(BACKGROUNDS)
    bg_map = {
        "black": (0, 0, 0), "dark blue": (0, 0, 50),
        "dark gray": (30, 30, 30), "navy": (0, 0, 80),
        "dark green": (0, 30, 0),
    }
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

    caption = _image_caption_from_params(shape_info, bg, rng)
    return img, caption


def _image_caption_from_params(shape_info, bg, rng):
    """Generate a Chinese caption from structured image parameters."""
    bg_zh = BG_ZH.get(bg, '黑色')

    if len(shape_info) == 1:
        s, c, sz = shape_info[0]
        s_zh = SHAPE_ZH.get(s, s)
        c_zh = COLOR_ZH.get(c, c)
        sz_zh = SIZE_ZH.get(sz, sz)

        bg_short = bg_zh.rstrip('色') if bg_zh.endswith('色') else bg_zh
        templates = [
            f"一个{c_zh}的{s_zh}在{bg_zh}背景上",
            f"图中有一个{c_zh}的{s_zh}",
            f"在{bg_short}色背景上画着一个{c_zh}的{s_zh}",
            f"画面中央是一个{c_zh}的{s_zh}",
            f"一个{sz_zh}的{c_zh}{s_zh}出现在{bg_zh}背景中",
            f"图片展示了一个{c_zh}的{s_zh}，背景为{bg_zh}",
            f"在{bg_short}色的背景上，有一个{c_zh}的{s_zh}",
            f"图像内容：{c_zh}的{s_zh}，{bg_zh}背景",
            f"这是一个{c_zh}的{s_zh}，位于{bg_zh}背景下",
            f"{c_zh}的{s_zh}形状物体，放在{bg_zh}底色上",
        ]
        return rng.choice(templates)

    elif len(shape_info) == 2:
        (s1, c1, _), (s2, c2, _) = shape_info
        s1_zh, s2_zh = SHAPE_ZH.get(s1, s1), SHAPE_ZH.get(s2, s2)
        c1_zh, c2_zh = COLOR_ZH.get(c1, c1), COLOR_ZH.get(c2, c2)

        bg_short = bg_zh.rstrip('色') if bg_zh.endswith('色') else bg_zh
        templates = [
            f"一个{c1_zh}{s1_zh}和一个{c2_zh}{s2_zh}在{bg_zh}背景上",
            f"{c1_zh}的{s1_zh}与{c2_zh}的{s2_zh}出现在{bg_zh}背景中",
            f"画面中有两个图形：{c1_zh}{s1_zh}和{c2_zh}{s2_zh}，背景为{bg_zh}",
            f"在{bg_short}色背景下，{c1_zh}{s1_zh}旁边是{c2_zh}{s2_zh}",
            f"图中左侧是{c1_zh}{s1_zh}，右侧是{c2_zh}{s2_zh}，背景{bg_zh}",
            f"{c1_zh}的{s1_zh}与{c2_zh}的{s2_zh}在{bg_short}色底色上",
            f"两个图形——{c1_zh}{s1_zh}和{c2_zh}{s2_zh}——位于{bg_zh}背景",
            f"图片有{c1_zh}{s1_zh}和{c2_zh}{s2_zh}，{bg_zh}为底色",
        ]
        return rng.choice(templates)

    else:
        parts = []
        for s, c, _ in shape_info:
            s_zh = SHAPE_ZH.get(s, s)
            c_zh = COLOR_ZH.get(c, c)
            parts.append(f"{c_zh}{s_zh}")

        templates = [
            f"画面中有{len(parts)}个图形：{'、'.join(parts)}，背景{bg_zh}",
            f"{'、'.join(parts)}出现在{bg_zh}背景中",
            f"在{bg_zh}背景下，排列着{'、'.join(parts)}",
            f"图片展示了{'、'.join(parts)}，底色为{bg_zh}",
            f"图像包含多个图形：{'、'.join(parts)}，{bg_zh}背景",
        ]
        return rng.choice(templates)


# ── Audio: Chinese-native caption generation ─────────────────────────

def _generate_audio_cn(seed=None):
    """Generate (mel_tensor[1,128,~125], Chinese caption)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Use same template pool as English but with Chinese descriptions
    idx = random.randrange(len(_AUDIO_GENERATORS_CN))
    waveform, caption = _AUDIO_GENERATORS_CN[idx]()

    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak * 0.5

    mel = waveform_to_mel(waveform)

    target_frames = N_FRAMES
    if mel.shape[-1] < target_frames:
        mel = torch.nn.functional.pad(mel, (0, target_frames - mel.shape[-1]))
    else:
        mel = mel[:, :, :target_frames]

    mel = mel.clamp(-3, 3) / 3.0
    return mel, caption


_AUDIO_GENERATORS_CN = []

def _make_audio_gen(waveform_fn, captions):
    """Create an audio generator with randomly selected Chinese caption."""
    return lambda wf=waveform_fn, caps=captions: (wf(), random.choice(caps))

# Pure tones
for freq, note in [(220, '低沉'), (440, '中频'), (880, '高频'), (1760, '极高')]:
    for vol, vol_desc in [(0.5, ''), (0.3, '轻柔的')]:
        caps = [
            f"一个{vol_desc}{note}的纯音信号",
            f"{vol_desc}{note}正弦波声音",
            f"一段{vol_desc}{note}的单一音调",
            f"频率为{freq}Hz的{vol_desc}纯音",
            f"{vol_desc}持续{note}鸣响",
            f"一个{vol_desc}{freq}赫兹的正弦波",
            f"喇叭播放{vol_desc}{note}纯音",
        ]
        _AUDIO_GENERATORS_CN.append(_make_audio_gen(
            lambda f=freq, a=vol: _sine_wave(f, amplitude=a), caps))

# Square waves
for freq, desc in [(110, '低沉'), (220, '嗡嗡'), (440, '刺耳')]:
    caps = [
        f"一个{desc}的方波声音",
        f"{desc}的电子合成音",
        f"一段{desc}的矩形波信号",
        f"{desc}的数码音效",
        f"类似{desc}电子乐的方波",
        f"一个{desc}的方形波音频",
    ]
    _AUDIO_GENERATORS_CN.append(_make_audio_gen(
        lambda f=freq: _square_wave(f), caps))

# White noise
_AUDIO_GENERATORS_CN.append(_make_audio_gen(
    lambda: _white_noise(),
    ["一段白噪音", "持续的静电噪音", "沙沙的白噪声", "均匀的白噪音信号",
     "一段随机的白噪声", "宽频白噪音"]))
_AUDIO_GENERATORS_CN.append(_make_audio_gen(
    lambda: _white_noise(amplitude=0.1),
    ["微弱的背景噪音", "很轻的背景噪声", "细微的底噪", "几乎听不见的静电声",
     "低音量的背景杂音"]))

# AM tones
for freq, rate in [(440, 4), (880, 8), (220, 2)]:
    speed_word = '快速' if rate >= 6 else '缓慢'
    caps = [
        f"一个{freq}Hz的脉冲调制音",
        f"有节奏的{freq}Hz颤音",
        f"一段周期性波动的{freq}Hz声音",
        f"被{rate}Hz调制的{freq}Hz载波音",
        f"{speed_word}脉动的{freq}Hz音调",
        f"一个以{rate}Hz速率波动的{freq}Hz声音",
    ]
    _AUDIO_GENERATORS_CN.append(_make_audio_gen(
        lambda f=freq, r=rate: _amplitude_modulated(freq_carrier=f, freq_mod=r), caps))

# FM tones
for base, dev in [(440, 200), (880, 300), (220, 100)]:
    caps = [
        "一个频率调制的颤音",
        "一段滑音效果的声音",
        "一个音高不断变化的调频音",
        "类似警报声的调频信号",
        "一段频率上下波动的音效",
        "一个音高周期性变化的合成音",
        "一段调频合成的声音",
    ]
    _AUDIO_GENERATORS_CN.append(_make_audio_gen(
        lambda b=base, d=dev: _frequency_modulated(freq_base=b, freq_dev=d), caps))

# Multi-tone
_AUDIO_GENERATORS_CN.append(_make_audio_gen(
    lambda: _sine_wave(440) + 0.5 * _sine_wave(880),
    ["两音合成的和弦", "双音叠加的和谐音程", "两个正弦波合成的音效",
     "440Hz和880Hz的叠加音", "八度音程的双音组合"]))
_AUDIO_GENERATORS_CN.append(_make_audio_gen(
    lambda: _sine_wave(440) + _square_wave(220),
    ["带有泛音的混合音色", "正弦波和方波的混合音", "音色丰富的合成声音",
     "基频加泛音的复合音"]))
_AUDIO_GENERATORS_CN.append(_make_audio_gen(
    lambda: _sine_wave(330) + 0.7 * _sine_wave(495) + 0.5 * _sine_wave(660),
    ["三音叠加的大三和弦", "三个音符的协和和弦", "大三和弦的合成音",
     "三音叠加的明亮和弦"]))
_AUDIO_GENERATORS_CN.append(_make_audio_gen(
    lambda: _sine_wave(262) + _sine_wave(330) + _sine_wave(392),
    ["C大调和弦的三个音符", "C大三和弦的分解音", "C-E-G三音和弦",
     "一个大调主和弦的合成"]))


# ── Video: Chinese-native caption generation ─────────────────────────

def _generate_video_cn(seed=None):
    """Generate ([3, T, H, W], Chinese caption) with native templates."""
    if seed is not None:
        random.seed(seed % 2**32)
        torch.random.manual_seed(seed % 2**32)

    shape_type = random.choice(['circle', 'square', 'triangle', 'star'])

    colors = {
        'red': (1.0, -1.0, -1.0), 'green': (-1.0, 1.0, -1.0),
        'blue': (-1.0, -1.0, 1.0), 'yellow': (1.0, 1.0, -1.0),
        'purple': (1.0, -1.0, 1.0), 'orange': (1.0, 0.5, -1.0),
        'cyan': (-1.0, 1.0, 1.0), 'white': (1.0, 1.0, 1.0),
    }
    color_name = random.choice(list(colors.keys()))
    color_rgb = colors[color_name]

    bg_colors = {
        'black': (-1.0, -1.0, -1.0), 'dark gray': (-0.7, -0.7, -0.7),
        'navy': (-1.0, -1.0, -0.5), 'dark green': (-1.0, -0.3, -1.0),
    }
    bg_name = random.choice(list(bg_colors.keys()))
    bg_rgb = bg_colors[bg_name]

    size_val = random.uniform(0.1, 0.2)

    directions = [
        ('left to right', 1.0, 0.0), ('right to left', -1.0, 0.0),
        ('top to bottom', 0.0, 1.0), ('bottom to top', 0.0, -1.0),
        ('diagonal', 0.8, 0.6), ('diagonal', -0.7, 0.7),
    ]
    dir_name, dx, dy = random.choice(directions)
    speed_val = random.uniform(0.15, 0.3)

    T, H, W = VIDEO_FRAMES, VIDEO_RESOLUTION, VIDEO_RESOLUTION
    video = torch.zeros(3, T, H, W)

    for t in range(T):
        progress = -1.0 + 2.0 * t / (T - 1) if T > 1 else 0.0
        cx = progress * speed_val * dx
        cy = progress * speed_val * dy
        offset_x = random.uniform(-0.3, 0.3)
        offset_y = random.uniform(-0.3, 0.3)
        cx += offset_x
        cy += offset_y
        cx = max(-0.8, min(0.8, cx))
        cy = max(-0.8, min(0.8, cy))

        frame = torch.zeros(3, H, W)
        for c in range(3):
            frame[c] = bg_rgb[c]
        frame = _draw_shape_on_canvas(frame, shape_type, color_rgb, cx, cy, size_val)
        video[:, t] = frame

    caption = _video_caption_from_params(shape_type, color_name, bg_name, dir_name, speed_val, random)
    return video, caption


def _video_caption_from_params(shape_type, color_name, bg_name, dir_name, speed, rng):
    """Generate a Chinese video caption from structured parameters."""
    s_zh = SHAPE_ZH.get(shape_type, shape_type)
    c_zh = COLOR_ZH.get(color_name, color_name)
    bg_zh = BG_ZH.get(bg_name, '黑色')
    d_zh = DIR_ZH.get(dir_name, dir_name)
    speed_desc = '快速' if speed > 0.25 else ('缓慢' if speed < 0.18 else '')

    bg_short = bg_zh.rstrip('色') if bg_zh.endswith('色') else bg_zh
    templates = [
        f"一个{c_zh}{s_zh}在{bg_zh}背景上{d_zh}移动",
        f"视频中，一个{c_zh}的{s_zh}从画面{d_zh}方向运动，背景{bg_zh}",
        f"{c_zh}的{s_zh}沿{d_zh}方向在{bg_short}色背景下移动",
        f"画面显示一个{c_zh}{s_zh}{d_zh}{speed_desc}滑动",
        f"在{bg_zh}背景中，有一个{c_zh}{s_zh}正沿{d_zh}运动",
        f"一个{c_zh}的{s_zh}物体在{bg_short}色底色上做{speed_desc}移动",
        f"片段呈现{c_zh}{s_zh}在{bg_zh}背景下沿{d_zh}平移",
        f"动画内容：{c_zh}的{s_zh}，方向{d_zh}，背景{bg_zh}",
        f"{speed_desc}{c_zh}{s_zh}在{bg_zh}背景上{d_zh}穿过画面",
        f"镜头中，一个{c_zh}{s_zh}以{d_zh}的方向划过{bg_zh}背景",
    ]
    return rng.choice(templates)


# ── Chinese Datasets ──────────────────────────────────────────────────

class ZhImageDataset(Dataset):
    """Image dataset with native Chinese captions."""
    def __init__(self, num_samples=10000, image_size=224, seed=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seed = (self.seed or 0) + idx + 42
        img, caption = _generate_image_cn(self.image_size, seed=seed)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        return img_tensor, caption


class ZhAudioDataset(Dataset):
    """Audio dataset with native Chinese captions."""
    def __init__(self, num_samples=5000, seed=None):
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mel, caption = _generate_audio_cn(seed=(self.seed or 0) + idx)
        return mel, caption


class ZhVideoDataset(Dataset):
    """Video dataset with native Chinese captions."""
    def __init__(self, num_samples=5000, seed=None):
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        vid, caption = _generate_video_cn(seed=(self.seed or 0) + idx)
        return vid, caption


# ── Test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Chinese Native Caption Templates ===\n")

    # Test image captions
    print("--- Image Captions ---")
    ds_img = ZhImageDataset(num_samples=10)
    for i in range(10):
        img, cap = ds_img[i]
        print(f"  [{i}] {cap}")

    print(f"\n--- Audio Captions ({len(_AUDIO_GENERATORS_CN)} templates) ---")
    ds_aud = ZhAudioDataset(num_samples=10)
    for i in range(10):
        mel, cap = ds_aud[i]
        print(f"  [{i}] {cap}")

    print("\n--- Video Captions ---")
    ds_vid = ZhVideoDataset(num_samples=10)
    for i in range(10):
        vid, cap = ds_vid[i]
        print(f"  [{i}] {cap}")

    # Count unique templates
    img_caps = set()
    for i in range(200):
        _, cap = ds_img[i]
        img_caps.add(cap)
    print(f"\nImage: {len(img_caps)} unique captions in 200 samples")

    aud_caps = set()
    for i in range(200):
        _, cap = ds_aud[i]
        aud_caps.add(cap)
    print(f"Audio: {len(aud_caps)} unique captions in 200 samples (from {len(_AUDIO_GENERATORS_CN)} templates × variations)")

    vid_caps = set()
    for i in range(200):
        _, cap = ds_vid[i]
        vid_caps.add(cap)
    print(f"Video: {len(vid_caps)} unique captions in 200 samples")

    print("\nChinese data generators: OK ✓")
