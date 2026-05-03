#!/usr/bin/env python3
"""
Minimal tokenizer — zero downloads, works offline.
Character-level BPE approximation that's good enough for MVP training.
Supports both English and Chinese text.
"""

import json
import re
from collections import Counter

VOCAB_SIZE = 10000


class SimpleTokenizer:
    """Simple byte-pair-like tokenizer. Built from scratch, no downloads.
    Supports both English and Chinese text.
    """

    def __init__(self, max_vocab=VOCAB_SIZE, add_chinese=True):
        self.max_vocab = max_vocab
        self._vocab = None
        self.inv_vocab = None
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        self.add_chinese = add_chinese
        self._build_vocab()

    def _get_cjk_chars(self):
        """Return essential Chinese characters needed for synthetic captions."""
        punct = list("，。！？；：’‘""（）【】、《》·—…")
        colors = list("红绿蓝黄紫橙青粉黑白灰棕金银")
        shapes = list("圆方三角星形心菱形")
        move = list("移动从到向上左右对角线背前景前方")
        desc = list("的一个小大有和含以及颜色包含个这在")
        nums = list("一二三四五六七八九十百千万亿")
        sound = list("声频音高低纯正弦波噪音柔和刺耳")
        other = list("纯这那不每张帧个图像视文")
        grammar = list("是着过了次第你我把他也都还可很能")
        extra = list("为与对中时后前上说下里外画面片长宽")
        more = list("使用制作生成显示变化模式类型种条线面及边粒状块信号深出现消失")
        addn = list("两三个同带与由被旋转")
        return list(set(punct + colors + shapes + move + desc + nums + sound + other + grammar + extra + more + addn))
        return list(set(punct + colors + shapes + move + desc + nums + sound + other + grammar + extra + more))
        return list(set(punct + colors + shapes + move + desc + nums + sound + other))

    def _build_vocab(self):
        """Build a simple vocab from English + Chinese characters."""
        chars = (
            list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
            + list("0123456789")
            + list(" .,!?;:\"'()-[]{}<>/@#$%^&*+=~`|_\\")
            + ["\n", "\t"]
        )

        specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<vid>"]
        self._vocab = {}
        for i, s in enumerate(specials):
            self._vocab[s] = i

        for c in chars:
            if c not in self._vocab:
                self._vocab[c] = len(self._vocab)

        if self.add_chinese:
            for c in self._get_cjk_chars():
                if c not in self._vocab and len(self._vocab) < self.max_vocab:
                    self._vocab[c] = len(self._vocab)

        common_ngrams = [
            "th", "he", "in", "er", "an", "on", "at", "en", "nd", "ti",
            "es", "or", "te", "of", "ed", "is", "it", "al", "ar", "le",
            "ing", "tion", "the", "and", "for", "are", "but", "not",
            "you", "all", "can", "had", "her", "was", "one", "our",
            "out", "has", "had", "hat", "the ", " a ", " an ", " in ",
            " on ", " at ", " to ", " of ", " is ", " it ", " as ",
            "moving", "moves", "across", "frame", "frames", "video",
            "circle", "square", "triangle", "shape", "color", "background",
            "small", "medium", "large", "with", "containing",
            "red", "blue", "green", "yellow", "purple", "orange",
            "pink", "white", "black", "navy", "dark", "gray",
        ]

        for ng in common_ngrams:
            if ng not in self._vocab and len(self._vocab) < self.max_vocab:
                self._vocab[ng] = len(self._vocab)

        for i in range(1000):
            if str(i) not in self._vocab and len(self._vocab) < self.max_vocab:
                self._vocab[str(i)] = len(self._vocab)

        self.inv_vocab = {v: k for k, v in self._vocab.items()}

    @property
    def vocab_size(self):
        return len(self._vocab) if self._vocab else 0

    def __len__(self):
        return self.vocab_size

    def encode(self, text):
        """Encode text to token IDs, longest match first."""
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for end in range(min(i + 15, len(text)), i, -1):
                substr = text[i:end]
                if substr in self._vocab:
                    tokens.append(self._vocab[substr])
                    i = end
                    matched = True
                    break
            if not matched:
                ch = text[i]
                if ch in self._vocab:
                    tokens.append(self._vocab[ch])
                else:
                    tokens.append(self.unk_token_id)
                i += 1
        return tokens

    def decode(self, ids):
        """Decode token IDs back to text."""
        if isinstance(ids, (int,)):
            ids = [ids]
        return "".join(self.inv_vocab.get(i, "<unk>") for i in ids)

    def __call__(self, texts, padding=False, truncation=False, max_length=None,
                 return_tensors=False):
        if isinstance(texts, str):
            texts = [texts]
        batch_ids = []
        for text in texts:
            ids = self.encode(text)
            if truncation and max_length:
                ids = ids[:max_length]
            if padding and max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
            batch_ids.append(ids)
        if return_tensors and padding:
            import torch
            return {"input_ids": torch.tensor(batch_ids),
                    "attention_mask": torch.tensor(
                        [[1 if i != self.pad_token_id else 0 for i in ids]
                         for ids in batch_ids])}
        return {"input_ids": batch_ids,
                "attention_mask": [[1] * len(ids) for ids in batch_ids]}


if __name__ == "__main__":
    tok = SimpleTokenizer()
    print(f"Vocabulary size: {tok.vocab_size}")

    text = "A small red circle on a black background."
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Encode: {text}")
    print(f"Decode: {decoded}")
    print(f"Match:  {text == decoded}")

    # Chinese test
    cn_text = "一个小红圆在黑色背景上"
    cn_ids = tok.encode(cn_text)
    cn_decoded = tok.decode(cn_ids)
    print(f"\nChinese: {cn_text}")
    print(f"Decode:  {cn_decoded}")
    print(f"Match:   {cn_text == cn_decoded}")

    batch = tok(["hello world", "test sentence"], padding=True, max_length=10, return_tensors=True)
    print(f"\nBatch input_ids: {batch['input_ids'].shape}")
    print("Tokenizer: OK ✓")
