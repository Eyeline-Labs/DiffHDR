import torch, math, csv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image

import torch, numpy as np
from PIL import Image

@torch.no_grad()
def _standardize(X: torch.Tensor, eps: float = 1e-6, dtype=torch.float32):
    # Always work in float32 for PCA math
    X = X.to(dtype)
    m = X.mean(0, keepdim=True)
    s = X.std (0, keepdim=True)
    return (X - m) / (s + eps), m, s

@torch.no_grad()
def _pca3(Xz: torch.Tensor):
    # Ensure float32 for pca_lowrank
    Xz = Xz.to(torch.float32)
    U, S, V = torch.pca_lowrank(Xz, q=3, center=False)  # V: [D,3], float32
    return V[:, :3]  # keep float32

@torch.no_grad()
def _to_rgb(img3: torch.Tensor, per_channel=True, percentile=1.0):
    # img3: [3,H,W] float32
    x = img3.detach().cpu().float()
    if percentile < 1.0:
        def pclip(a, p):
            lo, hi = np.percentile(a.flatten(), [(1-p)*50, 100-(1-p)*50])
            return np.clip(a, lo, hi)
        if per_channel:
            for c in range(3):
                x[c] = torch.from_numpy(pclip(x[c].numpy(), percentile))
        else:
            x = torch.from_numpy(pclip(x.numpy(), percentile))
    if per_channel:
        for c in range(3):
            ch = x[c]
            ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)
            x[c] = ch
    else:
        mn, mx = x.min(), x.max()
        x = (x - mn) / (mx - mn + 1e-6)
    x = (x.clamp(0,1) * 255.0 + 0.5).byte().numpy()
    return Image.fromarray(np.moveaxis(x, 0, 2), mode="RGB")

@torch.no_grad()
def vace_block_to_rgb_frames(
    block_feats: torch.Tensor,   # [B, C, T, H, W]
    batch_idx: int = 0,
    fit_stride_t: int = 1,
    fit_stride_spatial: int = 1,
    percentile: float = 0.995,
    per_channel_scale: bool = True
):
    B, C, T, H, W = block_feats.shape
    X = block_feats[min(batch_idx, B-1)].to(torch.float32)  # force float32 early

    # ---- Fit PCA basis (3 comps) on a subsampled set of tokens
    t_idx = torch.arange(0, T, max(1, fit_stride_t), device=X.device)
    h_idx = torch.arange(0, H, max(1, fit_stride_spatial), device=X.device)
    w_idx = torch.arange(0, W, max(1, fit_stride_spatial), device=X.device)

    X_flat = X.reshape(C, T*H*W).transpose(0,1)  # [T*H*W, C], float32
    sel = (t_idx[:,None,None]*H*W + h_idx[None,:,None]*W + w_idx[None,None,:]).reshape(-1)
    X_fit = X_flat[sel]                           # [N, C] float32
    Xz, _, _ = _standardize(X_fit, dtype=torch.float32)
    comps = _pca3(Xz)                             # [C,3] float32

    # ---- Project each frame and make RGB
    rgb_frames = []
    for t in range(T):
        Xt = X[:, t]                                   # [C,H,W] float32
        Xt_flat = Xt.reshape(C, H*W).transpose(0,1)    # [H*W,C]
        Xt_z, _, _ = _standardize(Xt_flat, dtype=torch.float32)
        Yt = Xt_z @ comps                              # [H*W,3] float32
        Yt = Yt.transpose(0,1).reshape(3, H, W)        # [3,H,W]
        rgb = _to_rgb(Yt, per_channel=per_channel_scale, percentile=percentile)
        rgb_frames.append(rgb)
    return rgb_frames


@torch.no_grad()
def save_vace_vis_rgb(
    vace_vis: list[torch.Tensor],
    block_idx: int = 0,
    outdir: str | Path = "vace_rgb",
    make_grid: bool = True,
    grid_cols: int = 9,
    **kwargs
):
    """
    vace_vis: list of [B,C,T,H,W] tensors (after unpatchify)
    Saves per-frame PNGs and an optional tiled grid.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    block = vace_vis[block_idx]
    frames = vace_block_to_rgb_frames(block, **kwargs)

    # Save individual frames
    for i, img in enumerate(frames):
        img.save(outdir / f"block{block_idx}_frame{i:04d}.png")

    # Optional grid
    if make_grid and len(frames) > 0:
        cols = grid_cols
        rows = math.ceil(len(frames) / cols)
        w, h = frames[0].size
        grid = Image.new("RGB", (cols*w, rows*h))
        for i, img in enumerate(frames):
            r, c = divmod(i, cols)
            grid.paste(img, (c*w, r*h))
        grid.save(outdir / f"block{block_idx}_grid.png")
    return frames
