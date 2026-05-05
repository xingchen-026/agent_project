#!/usr/bin/env python3
"""All dataset classes, image preprocessing, and collation — single source of truth.

Consolidates:
  7 PIL-to-tensor normalization copies → preprocess_image_path / preprocess_image_pil
  3 COCO dataset variants            → CocoCaptionDataset (unified)
  CocoCnDataset, VqaDataset          → from finetune_coco_cn / finetune_vqa
  CachedPairDataset                  → from train_clip
  6 collate variants                 → PadCollate (universal) + ImageCaptionCollate
  encode_captions                    → from train_joint_full
  interleave_loaders                 → from utils
  2 tensor_to_numpy copies           → single function
"""

import random
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from PIL import Image


# ── Image Preprocessing ─────────────────────────────────────────────

def preprocess_image_pil(image: Image.Image) -> torch.Tensor:
    """PIL.Image (already resized) → [C, H, W] tensor in [-1, 1]."""
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0


def preprocess_image_path(path: str, image_size: int = 224) -> torch.Tensor:
    """Open path → resize → tensor in [-1, 1]."""
    img = Image.open(path).convert('RGB')
    img = img.resize((image_size, image_size), Image.LANCZOS)
    return preprocess_image_pil(img)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """[-1, 1] tensor → [0, 1] numpy array (HWC)."""
    arr = ((tensor.detach().cpu().numpy() + 1) / 2).clip(0, 1)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    return arr


# ── COCO Caption Dataset (unified) ──────────────────────────────────

class CocoCaptionDataset(Dataset):
    """COCO images with English captions. Replaces 3 separate implementations.

    Parameters:
        coco_dir:       path to directory containing val2017/ (or other split)
        ann_file:       COCO annotation JSON file
        image_size:     resize target (default 224)
        max_images:     limit number of images (None = all)
        max_captions_per_image:  captions per image (default 3)
        pre_cache:      load all images into memory at init (True = fast, more RAM)
        seed:           random seed for image subset selection
    """

    def __init__(self, coco_dir, ann_file, image_size=224, max_images=None,
                 max_captions_per_image=3, pre_cache=True, seed=42):
        from pycocotools.coco import COCO

        coco_dir = Path(coco_dir)
        # Detect split directory (val2017, val2014, train2014, etc.)
        if (coco_dir / 'val2017').exists():
            self.img_dir = coco_dir / 'val2017'
        elif (coco_dir / 'val2014').exists():
            self.img_dir = coco_dir / 'val2014'
        elif (coco_dir / 'train2014').exists():
            self.img_dir = coco_dir / 'train2014'
        else:
            self.img_dir = coco_dir

        self.image_size = image_size
        self.pre_cache = pre_cache
        coco = COCO(str(ann_file))
        img_ids = sorted(coco.imgs.keys())

        if max_images:
            random.seed(seed)
            img_ids = random.sample(img_ids, min(max_images, len(img_ids)))

        self.samples = []
        self._cache = {} if pre_cache else None

        print(f"  Loading {len(img_ids)} COCO images...")
        for i, img_id in enumerate(img_ids):
            info = coco.imgs[img_id]
            path = self.img_dir / info['file_name']
            if not path.exists():
                continue

            if pre_cache:
                img = Image.open(path).convert('RGB')
                img = img.resize((image_size, image_size), Image.LANCZOS)
                self._cache[str(path)] = preprocess_image_pil(img)

            for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id))[:max_captions_per_image]:
                cap = ann['caption'].strip()
                if cap:
                    self.samples.append((str(path), cap))

            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{len(img_ids)}")

        print(f"  COCO: {len(self.samples)} pairs ({len(set(p for p, _ in self.samples))} images)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, caption = self.samples[idx]
        if self.pre_cache:
            return self._cache[path], caption
        return preprocess_image_path(path, self.image_size), caption


# ── COCO-CN Chinese Caption Dataset ─────────────────────────────────

class CocoCnDataset(Dataset):
    """COCO val2014/train2014 images with Chinese captions from COCO-CN."""

    def __init__(self, coco_dir, captions_file, image_size=224, max_samples=None):
        coco_dir = Path(coco_dir)
        img_dirs = {'val2014': coco_dir / 'val2014', 'train2014': coco_dir / 'train2014'}

        self.samples = []
        with open(captions_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '\t' not in line:
                    continue
                img_id, caption = line.split('\t', 1)
                img_name = img_id.split('#')[0]
                for img_dir in img_dirs.values():
                    img_path = img_dir / (img_name + '.jpg')
                    if img_path.exists():
                        self.samples.append((str(img_path), caption))
                        break

        if max_samples and len(self.samples) > max_samples:
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples)

        self.image_size = image_size
        val_count = sum(1 for p, _ in self.samples if 'val2014' in p)
        print(f"  COCO-CN: {len(self.samples)} pairs (val2014={val_count}, "
              f"train2014={len(self.samples) - val_count})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        return preprocess_image_path(img_path, self.image_size), caption


# ── Cached Pair Dataset ─────────────────────────────────────────────

class CachedPairDataset(Dataset):
    """Pre-cached image-text pairs with shared cache dictionary."""

    def __init__(self, pairs, image_size=224, cache=None):
        self.pairs = pairs
        if cache is not None:
            self._cache = cache
        else:
            unique_paths = sorted(set(p for p, _ in pairs))
            self._cache = {}
            print(f"  Pre-caching {len(unique_paths)} images...")
            for i, path in enumerate(unique_paths):
                self._cache[path] = preprocess_image_path(path, image_size)
                if (i + 1) % 1000 == 0:
                    print(f"    {i + 1}/{len(unique_paths)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, caption = self.pairs[idx]
        return self._cache[img_path], caption


# ── VQA Dataset ─────────────────────────────────────────────────────

VQA_TEMPLATES = [
    ("图片里有什么？", "图片里有{caption}"),
    ("描述这张图片", "{caption}"),
    ("图中是什么场景？", "{caption}"),
    ("这张照片展示了什么？", "这张照片展示了{caption}"),
    ("请描述图中的内容", "图中{caption}"),
    ("照片里有什么物体？", "照片里有{caption}"),
    ("画面中能看到什么？", "画面中能看到{caption}"),
    ("简单描述一下这张图", "{caption}"),
    ("图里有什么人/物？", "图里有{caption}"),
    ("这张图片的内容是什么？", "图片内容是{caption}"),
]


class VqaDataset(Dataset):
    """COCO images + templated Chinese VQA instruction pairs."""

    def __init__(self, coco_dir, captions_file, image_size=224, max_samples=None):
        coco_dir = Path(coco_dir)
        img_dirs = {'val2014': coco_dir / 'val2014', 'train2014': coco_dir / 'train2014'}

        self.samples = []
        with open(captions_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '\t' not in line:
                    continue
                img_id, caption = line.split('\t', 1)
                img_name = img_id.split('#')[0]
                for img_dir in img_dirs.values():
                    img_path = img_dir / (img_name + '.jpg')
                    if img_path.exists():
                        for question_t, answer_t in VQA_TEMPLATES:
                            answer = answer_t.format(caption=caption)
                            self.samples.append((str(img_path), question_t, answer))
                        break

        if max_samples and len(self.samples) > max_samples:
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples)

        self.image_size = image_size
        n_images = len(self.samples) // len(VQA_TEMPLATES)
        print(f"  VQA: {len(self.samples)} Q&A pairs ({n_images} images x {len(VQA_TEMPLATES)} templates)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, question, answer = self.samples[idx]
        text = f"问：{question}\n答：{answer}"
        return preprocess_image_path(img_path, self.image_size), text


# ── Collate Functions ───────────────────────────────────────────────

class PadCollate:
    """Universal collate: pad text, add BOS, return dict or tuple.

    Parameters:
        tokenizer:   SimpleTokenizer instance
        max_len:     max text length
        add_bos:     prepend <bos> token (id=2)
        return_dict: if True, return {'images':..., 'text_ids':..., 'attn_mask':...}
                     if False, return (images, text_ids, lengths)
        bos_id:      BOS token id (default 2)
    """

    def __init__(self, tokenizer, max_len=48, add_bos=True, return_dict=True, bos_id=2):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_bos = add_bos
        self.return_dict = return_dict
        self.bos_id = bos_id

    def __call__(self, batch):
        # batch: list of (tensor, str) or (tensor, str, str) for VQA
        images = torch.stack([item[0] for item in batch])
        texts = []
        for item in batch:
            if len(item) == 2:
                texts.append(item[1])
            else:
                # VQA: (image, question, answer) — combine
                texts.append(f"问：{item[1]}\n答：{item[2]}")
        bos_id = self.bos_id
        if self.add_bos:
            ids = [[bos_id] + self.tokenizer.encode(t)[:self.max_len - 1] for t in texts]
        else:
            ids = [self.tokenizer.encode(t)[:self.max_len] for t in texts]
        padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(i) for i in ids], batch_first=True, padding_value=0
        )
        if self.return_dict:
            attn_mask = (padded != 0).float()
            return {'images': images, 'text_ids': padded, 'attn_mask': attn_mask}
        else:
            return images, padded, [len(i) for i in ids]


class ImageCaptionCollate:
    """Simple collate: stack images, return raw captions as list."""

    def __call__(self, batch):
        imgs = torch.stack([item[0] for item in batch])
        captions = [item[1] for item in batch]
        return imgs, captions


# ── Encoding Helpers ────────────────────────────────────────────────

def encode_captions(tokenizer, captions, max_len=48, device='cuda', bos_id=2):
    """Encode captions with BOS token, return (padded_ids, lengths)."""
    ids = [[bos_id] + tokenizer.encode(c)[:max_len - 1] for c in captions]
    padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(i) for i in ids], batch_first=True, padding_value=0
    ).to(device)
    return padded, [len(i) for i in ids]


# ── Loader Utilities ────────────────────────────────────────────────

def interleave_loaders(*loaders):
    """Round-robin interleave multiple DataLoaders into a flat list of batches."""
    its = [iter(l) for l in loaders]
    done = [False] * len(its)
    result = []
    while not all(done):
        for i, it in enumerate(its):
            if not done[i]:
                try:
                    result.append(next(it))
                except StopIteration:
                    done[i] = True
    return result


def split_dataset(ds, val_frac=0.05, min_val=4, seed=42):
    """Random train/val split with minimum validation size."""
    n = len(ds)
    nv = max(min_val, int(n * val_frac))
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    return Subset(ds, idx[nv:]), Subset(ds, idx[:nv])
