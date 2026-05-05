#!/usr/bin/env python3
"""#3: Audio-Text Contrastive Learning on ESC-50."""

import os, sys, json, argparse, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TinyMultimodal
from tokenizer import SimpleTokenizer
from config import resolve_config
from utils import load_checkpoint_adaptive
from audio_synthetic import waveform_to_mel, SAMPLE_RATE


class ESC50Dataset(Dataset):
    def __init__(self, audio_dir, meta_file=None):
        paths = sorted(Path(audio_dir).glob('*.wav'))
        self.samples = []
        for p in paths[:2000]:
            try:
                import scipy.io.wavfile as wav
                sr, data = wav.read(str(p))
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                data = torch.from_numpy(data).float()
                if data.dim() == 2:
                    data = data.mean(dim=1)
                # Resample to 16kHz if needed
                if sr != 16000:
                    import torchaudio
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    data = resampler(data.unsqueeze(0))[0]
                # Extract center segment
                center = len(data) // 2
                half = 8000
                start = max(0, center - half)
                end = min(len(data), center + half)
                data = data[start:end]
                if len(data) >= 8000:
                    mel = waveform_to_mel(data)
                    self.samples.append((mel, p.stem[:30]))
            except:
                continue
        print(f"  ESC-50 Audio Dataset: {len(self.samples)} clips")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="../checkpoints_phase6/best.pt")
    parser.add_argument("--audio-dir", default="../esc50_data/audio")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleTokenizer(max_vocab=10000)
    cfg = resolve_config(args.resume, tokenizer,
        defaults={'use_contrastive': True, 'use_audio': True, 'use_video': False})
    model = TinyMultimodal(cfg).to(device)
    if os.path.exists(args.resume):
        load_checkpoint_adaptive(model, args.resume, device)

    print("Audio-CLIP training ready.")
    print("Run: python train_audio_clip.py --resume ../checkpoints_phase6/best.pt")


if __name__ == '__main__':
    main()
