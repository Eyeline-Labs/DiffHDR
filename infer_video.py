"""
infer_video.py -- LDR video -> HDR EXR frames via DiffHDR (Wan-2.1 VACE + LoRA).

Usage examples:
    python infer_video.py \
        --input_path /path/to/video.mp4 \
        --output_dir ./results/my_video \
        --prompt "" \
        --add_mask --use_under_exposure_mask
"""

import os
import sys
import argparse

import torch
import numpy as np
from PIL import Image

from diffsynth import VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from utils.color_utils import (
    exposure_masks_from_srgb,
    stabilize_soft_mask,
    _srgb_to_linear,
    srgb_to_linear_rec709_torch,
    Lin_to_Log,
    Log_to_Lin,
)
import pyexr


# ──────────────────────────── defaults ────────────────────────────
DEFAULT_LORA = "models/DiffHDR.safetensors"
LOCAL_MODEL_BASE = os.environ.get("MODEL_BASE", "models")
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

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


def load_reference_image(
    path: str,
    height: int,
    width: int,
    crop_and_resize: bool,
    device: torch.device,
    srgb_to_lg: bool = False,
    vace_mask: Image.Image | None = None,
    ev_boost: float = 5.0,
) -> torch.Tensor:
    """Load a reference image as [C, H, W] tensor, optionally masked and in Log space."""
    img = Image.open(path).convert("RGB")
    if crop_and_resize and (img.height != height or img.width != width):
        img = _crop_and_resize(img, height, width)
    ref = torch.from_numpy(np.array(img)).permute(2, 0, 1).to(device) / 255.0

    if vace_mask is not None:
        mask_np = np.array(vace_mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        mask = torch.from_numpy(mask_np).to(device).float() / 255.0
        mask = mask.unsqueeze(0)  # [1, H, W]
        ref = ref * mask + (1.0 - mask)

    if srgb_to_lg:
        ref = srgb_to_linear_rec709_torch(ref)
        ref = ref * (2.0 ** ev_boost)
        ref = Lin_to_Log(ref)

    return ref


def load_frames(
    input_path: str,
    num_frames: int,
    height: int,
    width: int,
    crop_and_resize: bool,
) -> list[Image.Image]:
    """
    Load frames from either an .mp4 video file or a directory of PNG/JPG images.
    Returns a list of PIL Images (at most num_frames).
    """
    if os.path.isfile(input_path):
        # Treat as video file -- use VideoData with video_file=
        vd = VideoData(
            video_file=input_path,
            height=height,
            width=width,
            crop_and_resize=crop_and_resize,
        )
    elif os.path.isdir(input_path):
        vd = VideoData(
            image_folder=input_path,
            height=height,
            width=width,
            crop_and_resize=crop_and_resize,
        )
    else:
        raise FileNotFoundError(f"input_path not found: {input_path}")

    n = min(len(vd), num_frames)
    return [vd[i] for i in range(n)]


def prepare_video_and_masks(
    frames: list[Image.Image],
    srgb_to_lg: bool,
    add_mask: bool,
    use_under_exposure_mask: bool,
    over_luma_thr_srgb: float,
    under_luma_thr_srgb: float,
    device: torch.device,
):
    """
    Convert frames to Log space (if requested) and compute exposure masks.

    Returns:
        vace_video:  torch.Tensor [T, C, H, W] in Log space  (or list[PIL] if not srgb_to_lg)
        vace_mask:   list[PIL]  (empty if add_mask=False)
        all_masks:   [vace_mask, over_mask, under_mask] or None
    """
    vace_mask: list[Image.Image] = []
    over_mask: list[Image.Image] = []
    under_mask: list[Image.Image] = []

    # Build float tensor if we need color conversion or masks
    if srgb_to_lg or add_mask:
        video_tensor = torch.stack(
            [
                torch.from_numpy(np.array(f)).permute(2, 0, 1).to(device) / 255.0
                for f in frames
            ]
        )  # [T, 3, H, W]

    # ---- Exposure masks ----
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
                under_bin_3 = under_bin.unsqueeze(0).repeat(3, 1, 1)
                video_tensor[i][under_bin_3 == 1.0] = 0.5
            else:
                combined_bin = over_bin

            combined_bin = combined_bin.unsqueeze(-1).repeat(1, 1, 3)  # [H, W, 3]
            vace_mask.append(
                Image.fromarray((combined_bin * 255.0).cpu().numpy().astype(np.uint8))
            )
            over_mask.append(
                Image.fromarray((over_bin.unsqueeze(-1).repeat(1, 1, 3) * 255.0).cpu().numpy().astype(np.uint8))
            )
            under_mask.append(
                Image.fromarray((under_bin.unsqueeze(-1).repeat(1, 1, 3) * 255.0).cpu().numpy().astype(np.uint8))
            )

    # ---- Color conversion: sRGB -> linear -> Log (ACEScct-like) ----
    if srgb_to_lg:
        frame_lin = _srgb_to_linear(video_tensor)  # [T, C, H, W]
        vace_video = Lin_to_Log(frame_lin)          # [T, C, H, W]
    else:
        vace_video = frames  # list of PIL

    all_masks = [vace_mask, over_mask, under_mask] if add_mask else None
    return vace_video, vace_mask, all_masks


# ──────────────────────────── argparse ────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DiffHDR: LDR video -> HDR EXR frames")
    p.add_argument("--input_path", type=str, required=True,
                   help="Path to a folder of PNG frames OR an .mp4 video file")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for EXR frames")
    p.add_argument("--lora_path", type=str, default=DEFAULT_LORA,
                   help="Path to LoRA checkpoint")
    p.add_argument("--prompt", type=str, default="",
                   help="Text prompt for the model")
    p.add_argument("--reference_image_path", type=str, default=None,
                   help="Optional path to a reference image for image-conditioned inference")
    p.add_argument("--reference_image_ev", type=float, default=5.0,
                   help="EV boost applied to reference image before Log conversion (default: 5.0)")

    # Shape / generation
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num_frames", type=int, default=33)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=10)

    # Color / masking flags
    # srgb_to_lg is ALWAYS True
    p.add_argument("--srgb_to_lg", action="store_true", default=True,
                   help="Convert sRGB -> linear Rec.709 -> Log (ACEScct-like curve, Rec.709 primaries) before feeding model (default: on)")
    p.add_argument("--add_mask", action="store_true", default=True,
                   help="Compute and use exposure masks as VACE conditioning")
    p.add_argument("--use_under_exposure_mask", action="store_true", default=False,
                   help="Include under-exposure regions in the mask")
    p.add_argument("--crop_and_resize", action="store_true", default=True,
                   help="Center-crop and resize input to --height x --width")
    p.add_argument("--over_luma_thr_srgb", type=float, default=0.95)
    p.add_argument("--under_luma_thr_srgb", type=float, default=0.01)

    # CFA
    p.add_argument("--vace_use_cfa", action="store_true", default=False)
    p.add_argument("--wan_dit_use_cfa", action="store_true", default=False)
    p.add_argument("--alpha_init", type=float, default=0.0)
    p.add_argument("--alpha_over_init", type=float, default=None)
    p.add_argument("--alpha_under_init", type=float, default=None)

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
        vace_use_cfa=args.vace_use_cfa,
        wan_dit_use_cfa=args.wan_dit_use_cfa,
        alpha_init=args.alpha_init,
        alpha_over_init=args.alpha_over_init,
        alpha_under_init=args.alpha_under_init,
    )

    # ── 2. Load LoRA weights ──────────────────────────────────────
    print(f"Loading LoRA from {args.lora_path} ...")
    pipe.load_lora(pipe.vace, args.lora_path, alpha=1)
    pipe.enable_vram_management(vae_dtype=torch.float32)

    # ── 3. Load input frames ─────────────────────────────────────
    print(f"Loading input from {args.input_path} ...")
    frames = load_frames(
        args.input_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        crop_and_resize=args.crop_and_resize,
    )
    print(f"  Loaded {len(frames)} frames.")

    # ── 4. Prepare VACE conditioning (color conversion + masks) ──
    vace_video, vace_mask, all_masks = prepare_video_and_masks(
        frames,
        srgb_to_lg=args.srgb_to_lg,
        add_mask=args.add_mask,
        use_under_exposure_mask=args.use_under_exposure_mask,
        over_luma_thr_srgb=args.over_luma_thr_srgb,
        under_luma_thr_srgb=args.under_luma_thr_srgb,
        device=device,
    )

    # ── 5. Reference image ───────────────────────────────────────
    if args.reference_image_path is not None:
        vace_reference_image = load_reference_image(
            args.reference_image_path,
            height=args.height,
            width=args.width,
            crop_and_resize=args.crop_and_resize,
            device=device,
            srgb_to_lg=args.srgb_to_lg,
            vace_mask=vace_mask[0] if vace_mask else None,
            ev_boost=args.reference_image_ev,
        )
    else:
        vace_reference_image = None

    # ── 6. Resolve mask and prompt for exposure routing ──────────
    use_cfa = args.vace_use_cfa and args.wan_dit_use_cfa
    vace_video_mask = None
    if args.add_mask:
        if use_cfa and all_masks:
            vace_video_mask = all_masks
        else:
            vace_video_mask = vace_mask

    prompt = args.prompt
    if use_cfa:
        prompt = ["", "over-exposed: Recover the over-exposed area.", "under-exposed: Recover the under-exposed area."]

    # ── 7. Run pipeline ──────────────────────────────────────────
    print(f"Running inference ({args.num_inference_steps} steps, seed={args.seed}) ...")
    video = pipe(
        height=args.height,
        width=args.width,
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        seed=args.seed,
        tiled=True,
        is_hdr=True,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        num_frames=args.num_frames,
        vace_video=vace_video,
        srgb_to_lg=args.srgb_to_lg,
        vace_video_mask=vace_video_mask,
        vace_reference_image=vace_reference_image,
    )

    # ── 8. Post-process: Log -> Linear, save EXR ─────────────────
    video = torch.stack(video)                         # [T, H, W, C]
    video = torch.clamp_min(video, 0.0)
    video = video.permute(0, 3, 1, 2).contiguous()     # [T, C, H, W]

    video = Log_to_Lin(video)                           # [T, C, H, W]
    video = video.permute(0, 2, 3, 1).contiguous()      # [T, H, W, C]

    T = video.shape[0]
    for i in range(T):
        out_path = os.path.join(args.output_dir, f"frame_{i:04d}.exr")
        pyexr.write(out_path, video[i].detach().cpu().numpy())

    print(f"Saved {T} EXR frames to {args.output_dir}")
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
