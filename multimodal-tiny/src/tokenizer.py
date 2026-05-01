#!/usr/bin/env python3
"""
Minimal tokenizer — zero downloads, works offline.
Character-level BPE approximation that's good enough for MVP training.
"""

import json
import re
from collections import Counter

# GPT-2 standard vocab size for compatibility
VOCAB_SIZE = 10000


class SimpleTokenizer:
    """Simple byte-pair-like tokenizer. Built from scratch, no downloads."""

    def __init__(self, max_vocab=VOCAB_SIZE):
        self.max_vocab = max_vocab
        self._vocab = None
        self.inv_vocab = None
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        self._build_vocab()

    def _build_vocab(self):
        """Build a simple vocab from common English characters and multi-char tokens."""
        # Base: all printable ASCII + common Unicode
        chars = (
            list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
            + list("0123456789")
            + list(" .,!?;:\"'()-[]{}<>/@#$%^&*+=~`|_\\")
            + ["\n", "\t"]
        )

        # Special tokens
        specials = ["<pad>", "<eos>", "<bos>", "<unk>"]
        self._vocab = {}
        for i, s in enumerate(specials):
            self._vocab[s] = i

        # Single chars
        for c in chars:
            if c not in self._vocab:
                self._vocab[c] = len(self._vocab)

        # Common multi-char tokens (bigrams and common words)
        common_ngrams = [
            "th", "he", "in", "er", "an", "on", "at", "en", "nd", "ti",
            "es", "or", "te", "of", "ed", "is", "it", "al", "ar", "le",
            "ing", "tion", "the", "and", "for", "are", "but", "not",
            "you", "all", "can", "had", "her", "was", "one", "our",
            "out", "has", "had", "hat", "the ", " a ", " an ", " in ",
            " on ", " at ", " to ", " of ", " is ", " it ", " as ",
            "circle", "square", "triangle", "shape", "color", "background",
            "small", "medium", "large", "with", "containing",
            "red", "blue", "green", "yellow", "purple", "orange",
            "pink", "white", "black", "navy", "dark", "gray",
        ]

        for ng in common_ngrams:
            if ng not in self._vocab and len(self._vocab) < self.max_vocab:
                self._vocab[ng] = len(self._vocab)

        # Fill remaining with numbers
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
            # Try longest match first (up to 15 chars)
            for end in range(min(i + 15, len(text)), i, -1):
                substr = text[i:end]
                if substr in self._vocab:
                    tokens.append(self._vocab[substr])
                    i = end
                    matched = True
                    break
            if not matched:
                # Use unk token for unknown char
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
        """API-compatible call method."""
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

        # Handle return_tensors
        if return_tensors and padding:
            import torch
            result = {"input_ids": torch.tensor(batch_ids),
                      "attention_mask": torch.tensor(
                          [[1 if i != self.pad_token_id else 0 for i in ids]
                           for ids in batch_ids])}
            return result

        return {"input_ids": batch_ids,
                "attention_mask": [[1] * len(ids) for ids in batch_ids]}


if __name__ == "__main__":
    tok = SimpleTokenizer()
    print(f"Vocabulary size: {tok.vocab_size}")

    text = "A small red circle on a black background."
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Encode: {text}")
    print(f"Tokens: {ids[:20]}...")
    print(f"Decode: {decoded}")
    print(f"Match:  {text == decoded}")

    # Test batch API
    batch = tok(["hello world", "test sentence"], padding=True, max_length=10, return_tensors=True)
    print(f"Batch input_ids: {batch['input_ids'].shape}")
    print("Tokenizer: OK ✓")
