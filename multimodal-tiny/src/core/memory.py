#!/usr/bin/env python3
"""Memory Bank + Diffusion Decoder — cross-modal compression + generation.

NOTE: When MemoryBank is active, the model's forward() compresses [video | image | audio]
sensory tokens into fixed-size memory tokens. The sequence layout changes to
[memory | text], which means position-based reconstruction heads (img_recon, aud_recon,
vid_recon) will slice wrong regions of the hidden states.

Workaround: set use_memory_bank=False when training reconstruction heads.
Set use_memory_bank=True for contrastive/diffusion training (uses memory_hidden).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.components import RMSNorm, SwiGLU


class MemoryBank(nn.Module):
    """Compresses sensory patches into fixed-size memory tokens via cross-attention."""
    def __init__(self, dim, n_mem=16, n_heads=8, mlp_mult=4):
        super().__init__()
        self.n_mem = n_mem
        self.memory = nn.Parameter(torch.empty(n_mem, dim))
        torch.nn.init.normal_(self.memory, std=0.02)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_mult)

    def forward(self, sensory):
        B = sensory.shape[0]
        mem = self.memory.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(mem, sensory, sensory)
        mem = self.norm1(mem + attn_out)
        mem = self.norm2(mem + self.mlp(mem))
        return mem


class DiffusionImageDecoder(nn.Module):
    """Latent diffusion decoder: denoises image patches conditioned on memory tokens.
    Uses DDIM sampling with 4-10 steps at inference.
    """
    def __init__(self, dim, num_patches=49, patch_dim=3072, latent_dim=256, num_steps=1000):
        super().__init__()
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.latent_dim = latent_dim
        self.num_steps = num_steps

        # Encoder: pixels → latent
        self.encoder = nn.Linear(patch_dim, latent_dim)
        # Decoder: latent → pixels
        self.decoder = nn.Linear(latent_dim, patch_dim)

        # Denoiser: latent + condition + timestep → predicted noise
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.cond_proj = nn.Linear(dim, latent_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(latent_dim * 2, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, latent_dim),
        )

        # Noise schedule (cosine)
        betas = self._cosine_schedule(num_steps)
        alphas = 1.0 - betas
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))

    def _cosine_schedule(self, steps, s=0.008):
        t = torch.linspace(0, steps, steps + 1)
        ft = torch.cos((t / steps + s) / (1 + s) * math.pi * 0.5) ** 2
        return torch.clamp(1 - ft[1:] / ft[:-1], 0, 0.999)

    def _encode(self, patches):
        return self.encoder(patches)  # [B, N, latent_dim]

    def _decode(self, latent):
        return self.decoder(latent)  # [B, N, patch_dim]

    def forward(self, patches, condition):
        """Training: add noise, predict noise. condition: [B, n_mem, dim] pooled."""
        B, N, _ = patches.shape
        z = self._encode(patches).view(B * N, -1)  # [B*N, latent_dim]
        cond = condition.mean(dim=1)  # [B, dim]

        # Sample timestep
        t = torch.randint(0, self.num_steps, (B,), device=patches.device)
        alpha_t = self.alphas_cumprod[t].view(B, 1, 1)

        # Add noise
        noise = torch.randn_like(z)
        z_noisy = torch.sqrt(alpha_t) * z.view(B, N, -1) + torch.sqrt(1 - alpha_t) * noise.view(B, N, -1)
        z_noisy = z_noisy.view(B * N, -1)

        # Predict noise
        t_embed = self.time_embed(t.float().view(-1, 1))  # [B, dim]
        cond_t = self.cond_proj(t_embed).unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        inp = torch.cat([z_noisy, cond_t], dim=-1)
        pred_noise = self.denoiser(inp)

        return F.mse_loss(pred_noise, noise), pred_noise

    @torch.no_grad()
    def sample(self, condition, num_inference_steps=10):
        """DDIM sampling: noise → image patches. condition: [B, n_mem, dim]."""
        B = condition.shape[0]
        cond = condition.mean(dim=1)  # [B, dim]
        device = condition.device

        z = torch.randn(B, self.num_patches, self.latent_dim, device=device)

        step_indices = torch.linspace(self.num_steps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        for i, t in enumerate(step_indices):
            t_batch = torch.full((B,), t, device=device)
            t_embed = self.time_embed(t_batch.float().view(-1, 1))
            cond_t = self.cond_proj(t_embed).unsqueeze(1).expand(B, self.num_patches, -1)

            inp = torch.cat([z.reshape(B * self.num_patches, -1),
                             cond_t.reshape(B * self.num_patches, -1)], dim=-1)
            pred_noise = self.denoiser(inp).view(B, self.num_patches, -1)

            alpha_t = self.alphas_cumprod[t]
            z = (z - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)

            if i < len(step_indices) - 1:
                t_next = step_indices[i + 1]
                alpha_next = self.alphas_cumprod[t_next]
                sigma = torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
                z = z + sigma * torch.randn_like(z)

        patches = self._decode(z)
        return patches
