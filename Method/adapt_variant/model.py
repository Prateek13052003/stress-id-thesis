"""
Physio-anchored, utilization-gated ADAPT variant.

Follows Mordacq et al. (ADAPT, MIDL 2024) with two changes:
  1. The anchor modality is configurable (train.py --anchor), defaulting to ECG
     instead of video (see data.py docstring for why; anchor_sweep.py compares
     all five choices under identical folds/hyperparameters).
  2. The multi-head self-attention is hand-rolled (not nn.TransformerEncoderLayer)
     so individual heads can be ablated at inference for interpretability.

Two-stage training, matching ADAPT section 3:
  - Anchoring: each non-anchor modality encoder is aligned to the frozen-after-
    stage-1 anchor encoder via symmetric InfoNCE, using only samples where BOTH
    the anchor and that modality are available.
  - Fusion: a small Transformer with [CLS] token fuses all modality embeddings.
    Missing modalities are masked out of attention (ADAPT eq. 2) instead of
    imputed, so the fusion stage requires no modality generator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, d_model=64, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        return self.net(x)


def info_nce(anchor_emb, other_emb, temperature=0.08):
    """Symmetric InfoNCE between paired (anchor, other) embeddings — ADAPT eq. 1."""
    a = F.normalize(anchor_emb, dim=-1)
    o = F.normalize(other_emb, dim=-1)
    logits_a2o = a @ o.T / temperature
    logits_o2a = o @ a.T / temperature
    targets = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits_a2o, targets) + F.cross_entropy(logits_o2a, targets)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, key_valid, ablate_heads=None):
        """
        x: (B, T, D) tokens ([CLS] + one per modality)
        key_valid: (B, T) bool, True = token is real (present modality or CLS)
        ablate_heads: optional iterable of head indices to zero out (interpretability)
        Returns: output (B, T, D), attn weights (B, H, T, T)
        """
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        key_mask = key_valid[:, None, None, :]
        scores = scores.masked_fill(~key_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)

        out = attn @ v
        if ablate_heads:
            out = out.clone()
            for h in ablate_heads:
                out[:, h, :, :] = 0.0
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out), attn


class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_ffn=256, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ffn, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_valid, ablate_heads=None):
        attn_out, attn_w = self.attn(x, key_valid, ablate_heads=ablate_heads)
        x = self.ln1(x + self.dropout(attn_out))
        x = self.ln2(x + self.dropout(self.ffn(x)))
        return x, attn_w


class MaskedMultimodalTransformer(nn.Module):
    def __init__(self, n_modalities, d_model=64, n_heads=4, n_layers=1, d_ffn=256, dropout=0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers)]
        )
        self.n_modalities = n_modalities

    def forward(self, modality_embs, avail, ablate_heads=None, ablate_layer=None):
        """
        modality_embs: (B, M, D) stacked, already projected to d_model
        avail: (B, M) bool availability mask
        Returns: cls_out (B, D), list of per-layer attention weights
        """
        B = modality_embs.size(0)
        cls_tok = self.cls.expand(B, -1, -1)
        x = torch.cat([cls_tok, modality_embs], dim=1)
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=avail.device)
        key_valid = torch.cat([cls_valid, avail], dim=1)

        attn_maps = []
        for li, block in enumerate(self.blocks):
            heads_to_ablate = ablate_heads if (ablate_layer is None or ablate_layer == li) else None
            x, attn_w = block(x, key_valid, ablate_heads=heads_to_ablate)
            attn_maps.append(attn_w)
        return x[:, 0, :], attn_maps


class PhysioAnchoredADAPT(nn.Module):
    def __init__(self, in_dims, n_classes=2, d_model=64, n_heads=4, n_layers=1,
                 d_ffn=256, dropout=0.1, encoder_hidden=64, encoder_dropout=0.3,
                 modalities=("ecg", "eda", "resp", "video", "audio")):
        super().__init__()
        self.modalities = list(modalities)
        self.encoders = nn.ModuleDict(
            {name: ModalityEncoder(in_dims[name], d_model, hidden=encoder_hidden, dropout=encoder_dropout)
             for name in self.modalities}
        )
        self.fusion = MaskedMultimodalTransformer(
            n_modalities=len(self.modalities), d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ffn=d_ffn, dropout=dropout,
        )
        self.classifier = nn.Linear(d_model, n_classes)

    def encode(self, feats):
        return {name: self.encoders[name](feats[name]) for name in self.modalities}

    def forward(self, feats, avail, ablate_heads=None, ablate_layer=None, ablate_modalities=None):
        """
        feats: dict[name] -> (B, in_dim)
        avail: (B, M) bool, column order == self.modalities
        ablate_modalities: optional iterable of modality names to force-mask out
          (used by the exact-Shapley utilization scorer in interpretability.py)
        """
        embs = self.encode(feats)
        stacked = torch.stack([embs[name] for name in self.modalities], dim=1)

        if ablate_modalities:
            avail = avail.clone()
            for name in ablate_modalities:
                avail[:, self.modalities.index(name)] = False

        cls_out, attn_maps = self.fusion(stacked, avail, ablate_heads=ablate_heads, ablate_layer=ablate_layer)
        logits = self.classifier(cls_out)
        return logits, attn_maps
