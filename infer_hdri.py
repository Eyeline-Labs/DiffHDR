"""
infer_hdri.py -- LDR panorama -> HDR EXR via VACE + HDRI LoRA.

Adapted from HDRI/examples/infer_hdri_wild.py for use with DiffHDR_Code pipeline.

Usage:
    python infer_hdri.py \
        --input_path /path/to/ldr_panorama.png \
        --output_dir ./results/hdri_test \
        --lora_path /path/to/hdri_lora.safetensors
"""

import os
import sys
import math
import argparse

import numpy as np
import pyexr
import torch
import torch.nn.functional as F
from PIL import Image

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


# ──────────────────────────── defaults ────────────────────────────
DEFAULT_LORA = "models/DiffHDR.safetensors"
LOCAL_MODEL_BASE = os.environ.get("MODEL_BASE", "models")


# ──────────────────────────── color utils ─────────────────────────
def _srgb_eotf(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    t = 0.04045
    return torch.where(x <= t, x / 12.92, torch.pow((torch.clamp(x, min=0) + a) / (1 + a), 2.4))


def _srgb_oetf(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    t = 0.0031308
    return torch.where(x <= t, 12.92 * x, (1 + a) * torch.pow(torch.clamp(x, min=t), 1 / 2.4) - a)


def _luma_709(rgb: torch.Tensor) -> torch.Tensor:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def _smoothstep(x, e0, e1):
    t = ((x - e0) / (e1 - e0 + 1e-8)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def lin_to_log(x: torch.Tensor, max_val=65536.0, gamma=2.2) -> torch.Tensor:
    x = torch.clamp(x, min=0.0, max=max_val)
    x = torch.log(gamma * x + 1.0) / math.log(gamma * max_val + 1.0)
    return torch.pow(x, 1.0 / gamma)


def log_to_lin(x: torch.Tensor, max_val=65536.0, gamma=2.2) -> torch.Tensor:
    x = torch.pow(x, gamma)
    x = torch.exp(x * math.log(gamma * max_val + 1.0)) - 1.0
    return x / gamma


def reinhard_tonemap(hdr_np, key=0.18, burn=1.0):
    lum = 0.2126 * hdr_np[..., 0] + 0.7152 * hdr_np[..., 1] + 0.0722 * hdr_np[..., 2]
    lum_mean = np.exp(np.mean(np.log(lum + 1e-8)))
    scaled = key / lum_mean * lum
    mapped = scaled * (1 + scaled / (burn * burn)) / (1 + scaled)
    ratio = (mapped / (lum + 1e-8))[..., None]
    return np.clip(hdr_np * ratio, 0, 1).astype(np.float32)


# ──────────────────────────── mask detection ──────────────────────
def make_overexposure_mask(srgb_chw, over_thr=0.95, no_morph=False):
    luma = _luma_709(_srgb_eotf(srgb_chw.clamp(0, 1)))
    luma_srgb = _srgb_oetf(luma.clamp(0, 1))
    over = _smoothstep(luma_srgb, over_thr - 0.02, over_thr + 0.02)
    ch_max = srgb_chw.max(dim=0).values
    ch_clip = _smoothstep(ch_max, 0.93, 0.99)
    over = torch.max(over, ch_clip)
    over_bin = (over > 0.2).float()

    if not no_morph:
        mask_4d = over_bin.unsqueeze(0).unsqueeze(0)
        # Morphological open (erode -> dilate)
        open_ks, open_pad = 5, 2
        mask_4d = -F.max_pool2d(-mask_4d, open_ks, stride=1, padding=open_pad)
        mask_4d = F.max_pool2d(mask_4d, open_ks, stride=1, padding=open_pad)
        # Morphological close (dilate -> erode)
        close_ks, close_pad = 7, 3
        mask_4d = F.max_pool2d(mask_4d, close_ks, stride=1, padding=close_pad)
        mask_4d = -F.max_pool2d(-mask_4d, close_ks, stride=1, padding=close_pad)
        over_bin = mask_4d.squeeze(0).squeeze(0)

    mask_3ch = over_bin.unsqueeze(-1).expand(-1, -1, 3)
    return Image.fromarray((mask_3ch * 255).numpy().astype(np.uint8))


# ──────────────────────────── argparse ────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="HDRI: LDR panorama -> HDR EXR")
    p.add_argument("--input_path", type=str, required=True,
                   help="Path to a single LDR panorama (PNG/JPG)")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--lora_path", type=str, default=DEFAULT_LORA)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--over_thr", type=float, default=0.95)
    p.add_argument("--no_mask", action="store_true", default=False,
                   help="Disable overexposure mask (all-white)")
    # This is ALWAYS True
    p.add_argument("--no_morph", action="store_true", default=True,
                   help="Disable morphological open/close on mask")
    p.add_argument("--prompt", type=str,
                   default="Restore the full dynamic range of this clipped HDRI panorama.")
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


# ──────────────────────────── main ────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if (not args.overwrite) and os.path.exists(os.path.join(args.output_dir, "predicted.exr")):
        print(f"[skip] Output exists in {args.output_dir}. Use --overwrite to re-run.")
        return

    # ── 1. Load pipeline ──────────────────────────────────────────
    print("Loading WanVideoPipeline ...")
    import glob as _glob
    _vace_dir = os.path.join(LOCAL_MODEL_BASE, "Wan-AI", "Wan2.1-VACE-14B")
    _dit_files = sorted(_glob.glob(os.path.join(_vace_dir, "diffusion_pytorch_model*.safetensors")))

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        vae_dtype=torch.float32,
        device="cuda",
        model_configs=[
            ModelConfig(path=_dit_files, offload_device="cpu"),
            ModelConfig(path=os.path.join(_vace_dir, "models_t5_umt5-xxl-enc-bf16.pth"), offload_device="cpu"),
            ModelConfig(path=os.path.join(_vace_dir, "Wan2.1_VAE.pth"), offload_device="cpu", offload_dtype=torch.float32),
        ],
        tokenizer_config=ModelConfig(path=os.path.join(LOCAL_MODEL_BASE, "Wan-AI", "Wan2.1-VACE-14B", "google", "umt5-xxl")),
        redirect_common_files=False,
    )

    # ── 2. Load LoRA weights ──────────────────────────────────────
    print(f"Loading LoRA from {args.lora_path} ...")
    pipe.load_lora(pipe.vace, args.lora_path, alpha=1)
    pipe.enable_vram_management(vae_dtype=torch.float32)

    # ── 3. Load input image ───────────────────────────────────────
    print(f"Loading input from {args.input_path} ...")
    img = Image.open(args.input_path).convert("RGB")
    img = img.resize((args.width, args.height), Image.LANCZOS)
    srgb = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)  # (3, H, W)

    # ── 4. Overexposure mask ──────────────────────────────────────
    if args.no_mask:
        mask_pil = Image.fromarray(np.ones((args.height, args.width, 3), dtype=np.uint8) * 255)
    else:
        mask_pil = make_overexposure_mask(srgb, over_thr=args.over_thr, no_morph=args.no_morph)

    # ── 5. VACE conditioning: sRGB -> linear -> log ──────────────
    srgb_lin = _srgb_eotf(srgb.clamp(0, 1))
    vace_frame = lin_to_log(srgb_lin)  # (3, H, W)

    # ── 6. Run pipeline ──────────────────────────────────────────
    print(f"Running inference ({args.num_inference_steps} steps, seed={args.seed}) ...")
    output = pipe(
        height=args.height,
        width=args.width,
        prompt=args.prompt,
        negative_prompt="",
        seed=args.seed,
        tiled=True,
        is_hdr=True,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        num_frames=1,
        vace_video=[vace_frame],
        srgb_to_lg=True,
        vace_video_mask=[mask_pil],
        vace_reference_image=None,
    )

    # ── 7. Post-process: log -> linear, save EXR ─────────────────
    out_log = torch.stack(output).permute(0, 3, 1, 2).contiguous().clamp(min=0)
    out_lin = log_to_lin(out_log).clamp(min=0)
    out_np = out_lin[0].permute(1, 2, 0).cpu().numpy()

    pyexr.write(os.path.join(args.output_dir, "predicted.exr"), out_np)

    # Save input LDR
    srgb_np = (srgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(srgb_np).save(os.path.join(args.output_dir, "input_ldr.png"))

    # Save mask
    mask_pil.save(os.path.join(args.output_dir, "mask.png"))

    # Save tonemapped preview
    tm = reinhard_tonemap(out_np)
    Image.fromarray((tm * 255).astype(np.uint8)).save(
        os.path.join(args.output_dir, "predicted_tonemapped.png"))

    print(f"Saved results to {args.output_dir}")
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
