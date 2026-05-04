#!/usr/bin/env python3
"""
VQA Inference Demo — test the instruction-tuned model interactively.
Usage:
  PYTHONIOENCODING=utf-8 python demo_vqa.py --checkpoint ../checkpoints_phase6_vqa/best.pt
"""

import os, sys, argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TinyMultimodal
from tokenizer import SimpleTokenizer
from config import resolve_config
from utils import load_checkpoint_adaptive


@torch.no_grad()
def generate_answer(model, tokenizer, image_tensor, question, device, max_len=64):
    """Generate answer from image + question."""
    prompt = f"问：{question}\n答："
    text_ids = torch.tensor([[2] + tokenizer.encode(prompt)], dtype=torch.long, device=device)
    text_ids = text_ids[:, :64]

    model.eval()
    gen_text = model.generate_text(image_tensor, tokenizer, max_len=max_len,
                                   temperature=0.7, top_k=30)
    # Extract answer part
    if '答：' in gen_text:
        answer = gen_text.split('答：', 1)[1].strip()
    else:
        answer = gen_text.strip()
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../checkpoints_phase6_vqa/best.pt")
    parser.add_argument("--image", type=str, default=None, help="Single image to query")
    parser.add_argument("--question", type=str, default="图片里有什么？")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleTokenizer(max_vocab=10000)
    cfg = resolve_config(args.checkpoint, tokenizer,
        defaults={'img_generation': True, 'use_audio': True, 'use_video': True})
    model = TinyMultimodal(cfg).to(device)
    load_checkpoint_adaptive(model, args.checkpoint, device)
    model.eval()

    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M ({cfg.describe()})")
    print(f"VQA Demo — type 'quit' to exit\n")

    # Use COCO-CN images if no specific image given
    coco_dir = Path("../coco_data")
    img_dirs = {'val2014': coco_dir / 'val2014', 'train2014': coco_dir / 'train2014'}

    # Collect available images
    sample_imgs = []
    for _, d in img_dirs.items():
        if d.exists():
            sample_imgs.extend(sorted(d.glob('*.jpg'))[:5])

    if args.image:
        sample_imgs = [Path(args.image)]

    test_questions = [
        "图片里有什么？",
        "描述这张图片",
        "图中是什么场景？",
        "照片里有什么物体？",
    ]

    for img_path in sample_imgs[:3]:
        print(f"=== {img_path.name} ===")
        img = Image.open(img_path).convert('RGB').resize((224, 224), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        img_tensor = img_tensor.to(device)

        for q in test_questions[:2]:
            answer = generate_answer(model, tokenizer, img_tensor, q, device)
            print(f"  Q: {q}")
            print(f"  A: {answer}")
        print()

    # Interactive mode if no specific image
    if not args.image and sample_imgs:
        print("Interactive mode — type question or 'q' to quit")
        img_path = sample_imgs[0]
        img = Image.open(img_path).convert('RGB').resize((224, 224), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        img_tensor = img_tensor.to(device)

        while True:
            try:
                q = input("\nQ: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ('q', 'quit', 'exit'):
                break
            if not q:
                continue
            answer = generate_answer(model, tokenizer, img_tensor, q, device)
            print(f"A: {answer}")


if __name__ == '__main__':
    main()
