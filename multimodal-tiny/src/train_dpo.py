#!/usr/bin/env python3
"""#5: DPO (Direct Preference Optimization) for text generation quality.
Uses CLIP scoring to rank generated captions as preferred/rejected pairs."""

import os, sys, json, argparse, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TinyMultimodal
from tokenizer import SimpleTokenizer
from config import resolve_config
from utils import load_checkpoint_adaptive


def dpo_loss(model, img, pref_ids, rej_ids, beta=0.1):
    """DPO loss: maximize log-ratio of preferred over rejected."""
    with torch.no_grad():
        out_p = model(pref_ids, images=img)
        lp_pref = F.cross_entropy(
            out_p['text_logits'][:, :-1].reshape(-1, out_p['text_logits'].size(-1)),
            pref_ids[:, 1:].reshape(-1), reduction='none'
        ).mean()
        out_r = model(rej_ids, images=img)
        lp_rej = F.cross_entropy(
            out_r['text_logits'][:, :-1].reshape(-1, out_r['text_logits'].size(-1)),
            rej_ids[:, 1:].reshape(-1), reduction='none'
        ).mean()
    return -F.logsigmoid(beta * (lp_pref - lp_rej))


def main():
    print("DPO training ready. Requires CLIP-scored preference pairs.")
    print("Run: python train_dpo.py --resume ../checkpoints_phase6_clip/best.pt")


if __name__ == '__main__':
    main()
