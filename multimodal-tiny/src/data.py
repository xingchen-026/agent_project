#!/usr/bin/env python3
"""
Data pipeline for Tiny Multimodal Model.
Downloads COCO data within 20GB budget and provides efficient DataLoader.
"""

import os
import json
import math
import random
import zipfile
from pathlib import Path
from io import BytesIO

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm


# ── Data budget constants ──────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BUDGET_GB = 20

# COCO URLs
COCO_URLS = {
    "val2017": {
        "zip": "http://images.cocodataset.org/zips/val2017.zip",
        "size_gb": 1.0,
    },
    "train2017": {
        "zip": "http://images.cocodataset.org/zips/train2017.zip",
        "size_gb": 19.0,  # full 118K images — we'll use a subset
    },
    "annotations": {
        "zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "size_gb": 0.25,
    },
}

# Training subset: use first N images from train2017
TRAIN_SUBSET_SIZE = 30000  # ~5GB of images
VAL_SIZE = 5000  # all of val2017


def ensure_coco_data(max_train_images=TRAIN_SUBSET_SIZE):
    """Download and prepare COCO data within budget.
    Returns dict with paths to annotations and image dirs.
    """
    import urllib.request

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ann_dir = DATA_DIR / "annotations"
    ann_dir.mkdir(exist_ok=True)
    val_dir = DATA_DIR / "val2017"
    val_dir.mkdir(exist_ok=True)
    train_dir = DATA_DIR / "train2017"
    train_dir.mkdir(exist_ok=True)

    # ── 1. Annotations (small, always download) ──
    ann_zip_path = DATA_DIR / "annotations_trainval2017.zip"
    val_ann = ann_dir / "captions_val2017.json"
    train_ann = ann_dir / "captions_train2017.json"

    if not (val_ann.exists() and train_ann.exists()):
        print("[data] Downloading annotations...")
        if not ann_zip_path.exists():
            try:
                _download_with_progress(COCO_URLS["annotations"]["zip"], ann_zip_path)
            except Exception as e:
                print(f"[data] Download failed: {e}")
                print("[data] Creating minimal annotation file for testing...")
                _create_minimal_annotation(val_ann, val_dir)
                _create_minimal_annotation(train_ann, train_dir, n=1000)
        if ann_zip_path.exists():
            with zipfile.ZipFile(ann_zip_path, 'r') as zf:
                for f in ['annotations/captions_val2017.json', 'annotations/captions_train2017.json']:
                    zf.extract(f, str(DATA_DIR))
            # Move from annotations/ subfolder
            for f in ['captions_val2017.json', 'captions_train2017.json']:
                src = DATA_DIR / "annotations" / f
                dst = ann_dir / f
                if src.exists() and not dst.exists():
                    src.rename(dst)
        print("[data] Annotations ready.")

    # ── 2. Val images (essential for eval) ──
    val_images = list(val_dir.glob("*.jpg"))
    if len(val_images) < 100:
        print("[data] Downloading val2017 images (~1GB)...")
        zip_path = DATA_DIR / "val2017.zip"
        _download_with_progress(COCO_URLS["val2017"]["zip"], zip_path)
        _extract_images(zip_path, val_dir, max_images=VAL_SIZE)
        print("[data] Val images ready.")

    # ── 3. Train images (subset within budget) ──
    train_images = list(train_dir.glob("*.jpg"))
    if len(train_images) < max_train_images:
        print(f"[data] Downloading train2017 images (subset: {max_train_images})...")
        zip_path = DATA_DIR / "train2017.zip"
        if not zip_path.exists():
            _download_with_progress(COCO_URLS["train2017"]["zip"], zip_path)
        _extract_images(zip_path, train_dir, max_images=max_train_images)
        print("[data] Train images ready.")

    # ── Report budget ──
    _report_budget()

    return {
        "val_ann": str(val_ann),
        "train_ann": str(train_ann),
        "val_dir": str(val_dir),
        "train_dir": str(train_dir),
    }


def _download_with_progress(url, dest):
    import urllib.request
    print(f"  Downloading {url.split('/')[-1]}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def _extract_images(zip_path, image_dir, max_images):
    """Extract up to max_images from a COCO zip into image_dir."""
    print(f"  Extracting up to {max_images} images...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = [m for m in zf.namelist() if m.endswith('.jpg')][:max_images]
        for member in tqdm(members, desc="  Extracting"):
            zf.extract(member, str(image_dir.parent))
    # Move files if nested in subfolder
    subdirs = list(image_dir.parent.glob("*/**/*.jpg"))
    if subdirs:
        import shutil
        for p in subdirs:
            if p.parent != image_dir:
                shutil.move(str(p), str(image_dir / p.name))
    print(f"  Extracted {len(list(image_dir.glob('*.jpg')))} images")


def _create_minimal_annotation(ann_file, img_dir, n=100):
    """Create a minimal annotation file for testing when download fails."""
    import datetime
    data = {
        "info": {"description": "Minimal test", "version": "1.0", "year": 2024,
                 "date_created": str(datetime.datetime.now())},
        "images": [],
        "annotations": [],
        "licenses": [{"id": 1, "name": "test"}],
    }
    for i in range(min(n, 5000)):
        data["images"].append({
            "id": i, "file_name": f"dummy_{i}.jpg",
            "width": 224, "height": 224,
        })
        data["annotations"].append({
            "id": i, "image_id": i,
            "caption": f"A test image number {i}.",
        })
    with open(ann_file, 'w') as f:
        json.dump(data, f)
    print(f"  Created minimal annotation: {ann_file}")


def _report_budget():
    total = sum(
        sum(f.stat().st_size for f in Path(root).rglob('*') if f.is_file())
        for root in [DATA_DIR]
    ) / (1024**3)
    print(f"[data] Total data size: {total:.2f} GB / {BUDGET_GB} GB budget")


# ── Dataset ────────────────────────────────────────────────────────────

class CocoDataset(Dataset):
    """COCO Captions dataset for multimodal training."""

    def __init__(self, ann_file, img_dir, max_samples=None, image_size=224,
                 tokenizer=None, max_text_len=48):
        self.img_dir = Path(img_dir)
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.tokenizer = tokenizer

        with open(ann_file) as f:
            data = json.load(f)

        # Build image_id -> file_name map
        img_map = {img["id"]: img["file_name"] for img in data["images"]}

        # Build samples: (img_file, caption)
        self.samples = []
        added = set()
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id in added:
                continue
            if img_id in img_map:
                fname = img_map[img_id]
                if (self.img_dir / fname).exists():
                    self.samples.append((fname, ann["caption"]))
                    added.add(img_id)
                    if max_samples and len(self.samples) >= max_samples:
                        break

        print(f"  Dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, caption = self.samples[idx]
        img_path = self.img_dir / fname
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0
        return img_tensor, caption


def collate_fn(batch, tokenizer, max_len):
    images, captions = zip(*batch)
    images = torch.stack(images)
    enc = tokenizer(
        list(captions), padding="max_length", truncation=True,
        max_length=max_len, return_tensors="pt",
    )
    return images, enc["input_ids"], enc["attention_mask"]


def build_loaders(tokenizer, config):
    """Build train/val DataLoaders."""
    paths = ensure_coco_data(max_train_images=config.get("train_size", 30000))

    val_ds = CocoDataset(
        paths["val_ann"], paths["val_dir"],
        max_samples=config.get("val_size", 500),
        image_size=config.get("image_size", 224),
        tokenizer=tokenizer,
        max_text_len=config.get("max_text_len", 48),
    )

    train_ds = CocoDataset(
        paths["train_ann"], paths["train_dir"],
        max_samples=config.get("train_size", 10000),
        image_size=config.get("image_size", 224),
        tokenizer=tokenizer,
        max_text_len=config.get("max_text_len", 48),
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.get("batch_size", 8),
        shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer, config.get("max_text_len", 48)),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.get("batch_size", 8),
        shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer, config.get("max_text_len", 48)),
        num_workers=0,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test data pipeline
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = {"train_size": 100, "val_size": 50, "batch_size": 4, "image_size": 224, "max_text_len": 48}
    train_loader, val_loader = build_loaders(tokenizer, config)
    for images, text_ids, mask in train_loader:
        print(f"Batch: images={images.shape}, text={text_ids.shape}")
        break
    print("Data pipeline: OK ✓")
