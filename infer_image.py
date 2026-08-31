"""
infer_image.py -- Single LDR image -> HDR EXR frames via DiffHDR (Wan-2.1 VACE + LoRA).

The single input image is replicated to num_frames before being fed through
the video diffusion pipeline.  Output is a sequence of EXR frames in linear
scene-referred HDR.

Usage:
    python infer_image.py \
        --input_path /path/to/photo.png \
        --output_dir ./results/single_image \
        --prompt ""
"""

import os
import argparse

import torch
import numpy as np
from PIL import Image

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from utils.color_utils import (
    exposure_masks_from_srgb,
    stabilize_soft_mask,
    _srgb_to_linear,
    Lin_to_Log,
    Log_to_Lin,
)
import pyexr


# ──────────────────────────── defaults ────────────────────────────
DEFAULT_LORA = "models/DiffHDR.safetensors"
LOCAL_MODEL_BASE = os.environ.get("MODEL_BASE", "models")
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


# ──────────────────── trained-config constants ──────────────────
TRAINED_HEIGHT = 720
TRAINED_WIDTH = 1280
TRAINED_NUM_FRAMES = 33
# DiT token count at trained resolution:
#   temporal = (33-1)//4+1 = 9, grid_h = 720//16 = 45, grid_w = 1280//16 = 80
#   total = 9 * 45 * 80 = 32400
_TRAINED_TOKENS = 9 * 45 * 80  # 32400


def compute_num_frames(height: int, width: int) -> int:
    """Pick num_frames (satisfying 4k+1) that best matches the trained token count."""
    grid_h = height // 16
    grid_w = width // 16
    spatial = grid_h * grid_w
    if spatial == 0:
        return TRAINED_NUM_FRAMES
    target_temporal = _TRAINED_TOKENS / spatial
    # temporal_tokens = (num_frames - 1) // 4 + 1  =>  num_frames = (t - 1) * 4 + 1
    best_f = int(np.ceil(target_temporal))
    best_f = max(best_f, 1)
    num_frames = (best_f - 1) * 4 + 1
    num_frames = max(num_frames, 1)
    return num_frames


# ──────────────────────────── helpers ─────────────────────────────
def _crop_and_resize(image: Image.Image, height: int, width: int) -> Image.Image:
    """Center-crop to target aspect ratio, then resize."""
    arr = np.array(image)
    ih, iw, _ = arr.shape
    if ih / iw < height / width:
        cw = int(ih / height * width)
        left = (iw - cw) // 2
        arr = arr[:, left : left + cw]
    else:
        ch = int(iw / width * height)
        top = (ih - ch) // 2
        arr = arr[top : top + ch, :]
    return Image.fromarray(arr).resize((width, height))


def load_and_replicate_image(
    image_path: str,
    num_frames: int,
    height: int,
    width: int,
    crop_and_resize: bool,
    device: torch.device,
    add_mask: bool,
    use_under_exposure_mask: bool,
    over_luma_thr_srgb: float,
    under_luma_thr_srgb: float,
):
    """
    Load a single image, replicate it to num_frames, convert to Log space,
    and compute exposure masks.

    Returns:
        vace_video:  torch.Tensor [T, C, H, W] in Log space
        vace_mask:   list[PIL]  (empty if add_mask=False)
    """
    img = Image.open(image_path).convert("RGB")
    if crop_and_resize and (img.height != height or img.width != width):
        img = _crop_and_resize(img, height, width)

    # Replicate the single image to create a "video"
    frames = [img] * num_frames

    # Build float tensor [T, 3, H, W]
    video_tensor = torch.stack(
        [
            torch.from_numpy(np.array(f)).permute(2, 0, 1).to(device) / 255.0
            for f in frames
        ]
    )

    # ---- Exposure masks ----
    vace_mask: list[Image.Image] = []
    if add_mask:
        prev_over_ema = prev_under_ema = None
        for i, frame_srgb in enumerate(video_tensor):
            over_soft, under_soft = exposure_masks_from_srgb(
                frame_srgb,
                over_luma_thr_srgb=over_luma_thr_srgb,
                under_luma_thr_srgb=under_luma_thr_srgb,
                softness=0.02,
                clip_eps_srgb=0.02,
                use_local_contrast_gate=False,
            )
            over_soft, prev_over_ema = stabilize_soft_mask(
                over_soft, prev_over_ema, alpha=0.7, k_smooth=9, k_open_close=9
            )
            under_soft, prev_under_ema = stabilize_soft_mask(
                under_soft, prev_under_ema, alpha=0.7, k_smooth=9, k_open_close=9
            )

            over_bin = (over_soft > 0.2).float()
            under_bin = (under_soft > 0.2).float()
            if use_under_exposure_mask:
                combined_bin = torch.logical_or(over_bin.bool(), under_bin.bool()).float()
            else:
                combined_bin = over_bin

            combined_bin = combined_bin.unsqueeze(-1).repeat(1, 1, 3)  # [H, W, 3]
            vace_mask.append(
                Image.fromarray((combined_bin * 255.0).cpu().numpy().astype(np.uint8))
            )

    # ---- sRGB -> linear -> Log (ACEScct-like) ----
    frame_lin = _srgb_to_linear(video_tensor)  # [T, C, H, W]
    vace_video = Lin_to_Log(frame_lin)          # [T, C, H, W]

    return vace_video, vace_mask


# ──────────────────────────── argparse ────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DiffHDR: single LDR image -> HDR EXR frames")
    p.add_argument("--input_path", type=str, required=True,
                   help="Path to a single PNG or JPG image")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for EXR frames")
    p.add_argument("--lora_path", type=str, default=DEFAULT_LORA,
                   help="Path to LoRA checkpoint")
    p.add_argument("--prompt", type=str, default="",
                   help="Text prompt for the model")

    # Shape / generation
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num_frames", type=int, default=None,
                   help="Number of frames (must be 4k+1). Auto-deduced from resolution if omitted.")
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=34)

    # Exposure mask
    p.add_argument("--use_under_exposure_mask", action="store_true", default=False,
                   help="Include under-exposure regions in the mask")
    p.add_argument("--over_luma_thr_srgb", type=float, default=0.95)
    p.add_argument("--under_luma_thr_srgb", type=float, default=0.01)

    # Preprocessing
    p.add_argument("--crop_and_resize", action="store_true", default=False,
                   help="Center-crop and resize input to --height x --width")

    # Misc
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Overwrite existing EXR outputs if they exist")
    return p.parse_args()


# ──────────────────────────── main ────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Skip if outputs already exist
    if (not args.overwrite) and os.path.exists(os.path.join(args.output_dir, "frame_0000.exr")):
        print(f"[skip] Output already exists in {args.output_dir}. Use --overwrite to re-run.")
        return

    device = torch.device("cuda")

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

    # ── 3. Resolve resolution and num_frames ────────────────────────
    if not args.crop_and_resize:
        img_probe = Image.open(args.input_path)
        args.width, args.height = img_probe.size  # PIL size is (W, H)
        args.height = args.height // 16 * 16
        args.width = args.width // 16 * 16
    if args.num_frames is None:
        args.num_frames = compute_num_frames(args.height, args.width)
    print(f"  Resolution: {args.height}x{args.width}, num_frames: {args.num_frames}")

    # ── 4. Load image, replicate, convert, compute masks ─────────
    print(f"Loading input image: {args.input_path}")
    vace_video, vace_mask = load_and_replicate_image(
        image_path=args.input_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        crop_and_resize=args.crop_and_resize,
        device=device,
        add_mask=True,
        use_under_exposure_mask=args.use_under_exposure_mask,
        over_luma_thr_srgb=args.over_luma_thr_srgb,
        under_luma_thr_srgb=args.under_luma_thr_srgb,
    )
    print(f"  Replicated to {args.num_frames} frames.  Masks computed: {len(vace_mask)}")

    # ── 5. Run pipeline ──────────────────────────────────────────
    print(f"Running inference ({args.num_inference_steps} steps, seed={args.seed}) ...")
    video = pipe(
        height=args.height,
        width=args.width,
        prompt=args.prompt,
        negative_prompt=NEGATIVE_PROMPT,
        seed=args.seed,
        tiled=True,
        is_hdr=True,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        num_frames=args.num_frames,
        vace_video=vace_video,
        srgb_to_lg=True,
        vace_video_mask=vace_mask if vace_mask else None,
        vace_reference_image=None,
    )

    # ── 6. Post-process: Log -> Linear, save EXR ─────────────────
    video = torch.stack(video)                         # [T, H, W, C]
    video = torch.clamp_min(video, 0.0)
    video = video.permute(0, 3, 1, 2).contiguous()     # [T, C, H, W]

    video = Log_to_Lin(video)                           # [T, C, H, W]
    video = video.permute(0, 2, 3, 1).contiguous()      # [T, H, W, C]

    # Save first frame only
    out_path = os.path.join(args.output_dir, f"frame_{0:04d}.exr")
    pyexr.write(out_path, video[0].detach().cpu().numpy())

    print(f"Saved 1 EXR frame to {args.output_dir}")
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
