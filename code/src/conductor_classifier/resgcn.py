"""Small ResGCN -- a residual network alternating spatial (graph) and temporal (1D conv) blocks.

Literature basis (docs/research/2026-08-07-skeleton-person-id-methods.md):
  * GCN-family models dominate skeleton-based person re-identification, and
    **a small ResGCN (645K params) outperformed the larger ST-GCN (3.3M
    params)** -- go smaller when data is scarcer.
  * Global average pooling over time and joints at the tail structurally
    guarantees **phase invariance**, while the temporal convolution attends
    to **frame order** (order carries the entire signal).
  * A 128-dim embedding combined with SupCon (a high-dimensional FC layer
    overfits at this small scale).
"""
import numpy as np
import torch
import torch.nn as nn


class SpatialGraphConv(nn.Module):
    """Mixes neighboring-joint information by multiplying the fixed adjacency matrix A by a learnable edge weight."""

    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)
        self.edge = nn.Parameter(torch.ones_like(A))
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):                      # x: (B,C,T,V)
        x = self.conv(x)
        return torch.einsum("bctv,vw->bctw", x, self.A * self.edge)


class ResBlock(nn.Module):
    """Spatial graph conv -> temporal 1D conv (k=9) + residual. stride reduces the time axis."""

    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor, stride: int = 1,
                 k: int = 9, dropout: float = 0.1):
        super().__init__()
        self.gcn = SpatialGraphConv(in_ch, out_ch, A)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.tcn = nn.Conv2d(out_ch, out_ch, kernel_size=(k, 1),
                             padding=(k // 2, 0), stride=(stride, 1))
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)
        if in_ch == out_ch and stride == 1:
            self.short = nn.Identity()
        else:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        res = self.short(x)
        y = self.act(self.bn1(self.gcn(x)))
        y = self.drop(self.bn2(self.tcn(y)))
        return self.act(y + res)


class ResGCN(nn.Module):
    """(B,C,T,V) -> logits (B,num_classes). If return_embedding=True, returns (logits, 128-dim embedding)."""

    def __init__(self, in_channels: int, num_classes: int,
                 adjacency: np.ndarray, embed_dim: int = 128,
                 widths=(32, 32, 64, 64, 96, 96), strides=(1, 1, 2, 1, 2, 1)):
        super().__init__()
        A = torch.as_tensor(adjacency, dtype=torch.float32)
        self.data_bn = nn.BatchNorm1d(in_channels * A.shape[0])
        self.strides = tuple(strides)
        blocks, ch = [], in_channels
        for w, s in zip(widths, strides):
            blocks.append(ResBlock(ch, w, A, stride=s))
            ch = w
        self.blocks = nn.ModuleList(blocks)
        self.embed = nn.Linear(ch, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def features(self, x):
        """Feature map **before** pooling, (B,C',T',V) -- the point where temporal structure is still intact."""
        b, c, t, v = x.shape
        x = self.data_bn(x.permute(0, 1, 3, 2).reshape(b, c * v, t))
        x = x.reshape(b, c, v, t).permute(0, 1, 3, 2).contiguous()
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x, return_embedding: bool = False, mask=None):
        """mask: (B,T), 1=actually observed frame / 0=padding. When given, the
        global average pooling **excludes padded frames from the denominator**
        (dividing by the actual frame count rather than the window length).
        The mask is downsampled via nearest-neighbor to match however much the
        blocks' stride has shrunk the time axis. Since the average is the
        window's final decision, padded frames lose their vote at this point
        (this does not block them from the intermediate convolutions -- zeros
        at window boundaries already existed there anyway)."""
        x = self.features(x)
        if mask is None:
            x = x.mean(dim=(2, 3))      # global average pooling over time and joints -> cancels out phase
        else:
            m = mask.to(x.dtype)[:, None, :]                    # (B,1,T)
            m = torch.nn.functional.interpolate(m, size=x.shape[2], mode="nearest")
            m = m[:, :, :, None]                                # (B,1,T',1)
            denom = m.sum(dim=(2, 3)).clamp(min=1.0) * x.shape[3]
            x = (x * m).sum(dim=(2, 3)) / denom
        emb = self.embed(x)
        logit = self.fc(emb)
        return (logit, emb) if return_embedding else logit
