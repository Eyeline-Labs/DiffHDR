import torch
from einops import rearrange

import os
import numpy as np
import torch.nn.functional as F
from PIL import Image

# --- helper: build 64ch latent mask exactly like vace_mask_latents ---
def build_mask_latents(mask_video_3chw: torch.Tensor) -> torch.Tensor:
    """
    mask_video_3chw: [1, 3, T, H_img, W_img] in [0,1] (after preprocess)
    returns: [1, 64, T_lat, H_lat, W_lat] (same layout as vace_mask_latents)
    """
    m = mask_video_3chw[0, 0]  # [T, H_img, W_img]
    m_lat = rearrange(m, "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)  # [1,64,T,H_lat,W_lat]
    m_lat = torch.nn.functional.interpolate(
        m_lat,
        size=((m_lat.shape[2] + 3) // 4, m_lat.shape[3], m_lat.shape[4]),
        mode="nearest-exact",
    )
    return m_lat

# --- helper: convert 64ch latent mask -> token gate [B,S,1] aligned with patch_size=(1,2,2) ---
def mask_latents_to_gate(mask_latents_64: torch.Tensor) -> torch.Tensor:
    """
    mask_latents_64: [B,64,T,H,W] in [0,1]
    returns: [B,S,1] where S = T*(H/2)*(W/2)
    """
    # collapse 64ch to 1ch (no new params)
    m = mask_latents_64.mean(dim=1, keepdim=True)  # [B,1,T,H,W]
    # downsample to patch grid (1,2,2) to align with vace_patch_embedding
    m = torch.nn.functional.avg_pool3d(m, kernel_size=(1, 2, 2), stride=(1, 2, 2))  # [B,1,T,H/2,W/2]
    g = m.flatten(2).transpose(1, 2)  # [B,S,1]
    return g.clamp(0, 1)


def _save_gray_png_from_tensor(hw: torch.Tensor, path: str):
    """
    hw: [H, W] torch tensor, values in [0,1] (float/half ok)
    """
    arr = (hw.detach().float().clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)

def debug_save_token_gate_pil(
    token_gate: torch.Tensor,      # [B, S, 1]
    mask_latents_64: torch.Tensor, # [B, 64, T, H_lat, W_lat]  (the "over_lat" you built)
    raw_mask_video: torch.Tensor,  # [B, 3, T, H_img, W_img] or None (preprocessed mask video)
    out_dir: str,
    prefix: str,
    max_frames: int = 8,
    save_raw: bool = True,
    save_upsampled: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    B, S, _ = token_gate.shape
    _, _, T, H_lat, W_lat = mask_latents_64.shape
    Ht, Wt = H_lat // 2, W_lat // 2  # because your patch_size=(1,2,2)

    expected_S = T * Ht * Wt
    assert S == expected_S, f"S mismatch: S={S}, expected {expected_S} (T={T},Ht={Ht},Wt={Wt})"

    # [B,S,1] -> [B,T,Ht,Wt]
    gate_grid = token_gate.view(B, T, Ht, Wt)

    # Decide target resolution for visualization
    if raw_mask_video is not None:
        H_img, W_img = raw_mask_video.shape[-2], raw_mask_video.shape[-1]
    else:
        # fallback: visualize at (Ht,Wt)
        H_img, W_img = Ht, Wt

    # Upsample to image size for easier inspection: [B,T,H_img,W_img]
    if save_upsampled:
        gate_up = F.interpolate(
            gate_grid.reshape(B * T, 1, Ht, Wt),
            size=(H_img, W_img),
            mode="nearest",
        ).reshape(B, T, H_img, W_img)
    print(T, max_frames)
    t_save = min(T, max_frames)
    print(t_save)
    for t in range(t_save):
        if save_raw and (raw_mask_video is not None):
            # raw mask (channel 0)
            _save_gray_png_from_tensor(
                raw_mask_video[0, 0, t],
                os.path.join(out_dir, f"{prefix}_raw_t{t:03d}.png"),
            )

        if save_upsampled:
            _save_gray_png_from_tensor(
                gate_up[0, t],
                os.path.join(out_dir, f"{prefix}_gate_t{t:03d}.png"),
            )
        # else:
        # save at token grid resolution
        _save_gray_png_from_tensor(
            gate_grid[0, t],
            os.path.join(out_dir, f"{prefix}_gategrid_t{t:03d}.png"),
        )
