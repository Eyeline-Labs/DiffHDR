"""
Long-video HDR inference with sliding-window temporal blending.

Processes a folder of PNG frames (hundreds or thousands) through the
WanVideoPipeline with LoRA, producing per-frame EXR outputs in linear HDR.

The algorithm uses a fixed-size sliding window (default 33 frames, stride 16)
with linear blend ramps in the overlap region, and a streaming buffer that
writes completed frames to disk as soon as they are fully blended, keeping
memory usage O(window_size) rather than O(total_frames).

Usage:
    python infer_long_video.py \
        --input_path /path/to/png_frames/ \
        --output_dir /path/to/output/ \
        --prompt "" \
        --add_mask --crop_and_resize --srgb_to_lg
"""

import argparse
import os
import sys

import numpy as np
import pyexr
import torch
from PIL import Image

from diffsynth import VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

from utils.color_utils import (
    ColourUtils,
    _srgb_to_linear,
    exposure_masks_from_srgb,
    Lin_to_Log,
    Log_to_Lin,
    stabilize_soft_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_LORA_PATH = "models/DiffHDR.safetensors"
LOCAL_MODEL_BASE = os.environ.get("MODEL_BASE", "models")

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def _natural_sort_key(fn: str):
    """Sort key for frame names like 0001.png, 0002.png, frame_0010.png."""
    base, _ = os.path.splitext(fn)
    try:
        return (0, int(base))
    except ValueError:
        # Fall back: try to extract trailing digits
        digits = ""
        for ch in reversed(base):
            if ch.isdigit():
                digits = ch + digits
            else:
                break
        if digits:
            return (0, int(digits))
        return (1, base)


def build_1d_mask(
    length: int,
    left_bound: bool,
    right_bound: bool,
    border_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create 1-D blend mask with linear ramps at boundaries.

    - left_bound=True  -> no fade-in  at start (weight=1 from frame 0)
    - right_bound=True -> no fade-out at end   (weight=1 to last frame)
    - Otherwise: linear ramp 0->1 over ``border_width`` frames at each end.

    Returns a 1-D tensor of shape ``(length,)``.
    """
    x = torch.ones((length,), device=device, dtype=dtype)
    if border_width == 0:
        return x
    shift = 0.5
    if not left_bound:
        x[:border_width] = (
            torch.arange(border_width, device=device, dtype=dtype) + shift
        ) / border_width
    if not right_bound:
        x[-border_width:] = torch.flip(
            (torch.arange(border_width, device=device, dtype=dtype) + shift)
            / border_width,
            dims=(0,),
        )
    return x


def _crop_and_resize(image: Image.Image, height: int, width: int) -> Image.Image:
    image_np = np.array(image)
    image_height, image_width, _ = image_np.shape
    if image_height / image_width < height / width:
        cropped_width = int(image_height / height * width)
        left = (image_width - cropped_width) // 2
        image_np = image_np[:, left : left + cropped_width]
        image = Image.fromarray(image_np).resize((width, height))
    else:
        cropped_height = int(image_width / width * height)
        top = (image_height - cropped_height) // 2
        image_np = image_np[top : top + cropped_height, :]
        image = Image.fromarray(image_np).resize((width, height))
    return image


def load_image(
    image_path: str,
    height: int,
    width: int,
    crop_and_resize: bool,
    colour_utils: ColourUtils,
) -> torch.Tensor:
    """Load a single image as a [C, H, W] tensor in [0, 1]."""
    image = Image.open(image_path).convert("RGB")
    if crop_and_resize and (image.height != height or image.width != width):
        image = _crop_and_resize(image, height, width)
    image = (
        torch.from_numpy(np.array(image))
        .permute(2, 0, 1)
        .to(colour_utils.device)
        / 255.0
    )
    return image


def load_video_slice(
    image_folder: str,
    file_list: list[str],
    height: int,
    width: int,
    crop_and_resize: bool,
    srgb_to_lg: bool,
    add_mask: bool,
    use_under_exposure_mask: bool,
    over_luma_thr_srgb: float,
    under_luma_thr_srgb: float,
    colour_utils: ColourUtils,
):
    """Load a window of frames by ``file_list`` and apply colour / mask transforms.

    Returns:
        video_out:  If srgb_to_lg, a [T, C, H, W] tensor in Log space.
                    Otherwise a list of PIL images.
        vace_mask:  List of PIL mask images (or empty list).
        all_masks:  [vace_mask, over_mask, under_mask] or None.
    """
    video = VideoData(
        image_folder=image_folder,
        file_list=file_list,
        height=height,
        width=width,
        crop_and_resize=crop_and_resize,
    )
    video = [video[i] for i in range(len(video))]

    vace_mask: list[Image.Image] = []
    over_mask: list[Image.Image] = []
    under_mask: list[Image.Image] = []

    video_tensor = None
    if srgb_to_lg or add_mask:
        video_tensor = torch.stack(
            [
                torch.from_numpy(np.array(frame))
                .permute(2, 0, 1)
                .to(colour_utils.device)
                / 255.0
                for frame in video
            ]
        )  # [T, C, H, W]

    if add_mask:
        prev_over_ema = None
        prev_under_ema = None
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
                combined_bin = torch.logical_or(
                    over_bin.bool(), under_bin.bool()
                ).float()
            else:
                combined_bin = over_bin

            # Paint under-exposed pixels to mid-gray so the model knows they
            # need reconstruction.
            if use_under_exposure_mask:
                under_bin_3 = under_bin.unsqueeze(0).repeat(3, 1, 1)
                video_tensor[i][under_bin_3 == 1.0] = 0.5

            combined_bin_hwc = combined_bin.unsqueeze(-1).repeat(1, 1, 3)  # [H, W, 3]
            vace_mask.append(
                Image.fromarray(
                    (combined_bin_hwc * 255.0).cpu().numpy().astype(np.uint8)
                )
            )
            over_mask.append(
                Image.fromarray(
                    (over_bin.unsqueeze(-1).repeat(1, 1, 3) * 255.0)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
            )
            under_mask.append(
                Image.fromarray(
                    (under_bin.unsqueeze(-1).repeat(1, 1, 3) * 255.0)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
            )

    if srgb_to_lg:
        frame_lin = _srgb_to_linear(video_tensor)  # [T, C, H, W]
        frame_log = Lin_to_Log(frame_lin)
        video_out = frame_log  # torch tensor
    else:
        video_out = video  # list of PIL

    all_masks = [vace_mask, over_mask, under_mask] if add_mask else None
    return video_out, vace_mask, all_masks


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Long-video HDR inference with sliding-window blending."
    )

    # I/O
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Folder containing PNG frames (sorted naturally).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for EXR frames.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=DEFAULT_LORA_PATH,
        help="Path to LoRA checkpoint.",
    )

    # Prompt
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt for the video.",
    )

    # Sliding window
    parser.add_argument(
        "--window_size",
        type=int,
        default=33,
        help="Number of frames per sliding window.",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=16,
        help="Step between window starts.",
    )
    parser.add_argument(
        "--use_prev_window_reference",
        action="store_true",
        default=False,
        help="Use previous window's output frame as reference for next window (temporal consistency).",
    )

    # Spatial
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)

    # Diffusion
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=34)

    # Colour pipeline. This is ALWAYS True
    parser.add_argument(
        "--srgb_to_lg",
        action="store_true",
        default=True,
        help="Convert sRGB -> linear Rec.709 -> Log (lg) for model input (default: True).",
    )
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit total frames processed (default: all).")
    parser.add_argument("--add_mask", action="store_true", default=False)
    parser.add_argument("--use_under_exposure_mask", action="store_true", default=False)
    parser.add_argument("--crop_and_resize", action="store_true", default=False)
    parser.add_argument("--over_luma_thr_srgb", type=float, default=0.95)
    parser.add_argument("--under_luma_thr_srgb", type=float, default=0.01)

    # Misc
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing EXR output.",
    )

    # Multi-GPU distribution
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank of this process (0..world_size-1).",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Total number of processes.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Discover input frames
    # ------------------------------------------------------------------
    if not os.path.isdir(args.input_path):
        raise FileNotFoundError(f"Input path does not exist: {args.input_path}")

    frame_names = [
        f
        for f in os.listdir(args.input_path)
        if f.lower().endswith(".png")
    ]
    if not frame_names:
        raise RuntimeError(f"No PNG frames found in {args.input_path}")

    frame_names = sorted(frame_names, key=_natural_sort_key)
    if args.max_frames is not None:
        frame_names = frame_names[: args.max_frames]
    T_total = len(frame_names)
    print(f"Found {T_total} frames in {args.input_path}")
    print(
        f"Sliding window: size={args.window_size}, stride={args.window_stride}"
    )
    if args.use_prev_window_reference:
        print(
            "Temporal consistency: using previous window's frame at index "
            "window_stride as reference for the next window."
        )

    # ------------------------------------------------------------------
    # 2. Check for existing output / create output directory
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    first_exr = os.path.join(args.output_dir, "frame_0001.exr")
    if (not args.overwrite) and os.path.exists(first_exr):
        print(f"[skip] Output already exists at {args.output_dir} (use --overwrite to force)")
        return

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    colour_utils = ColourUtils()

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
    print(f"Loading LoRA from {args.lora_path}")
    pipe.load_lora(pipe.vace, args.lora_path, alpha=1)
    pipe.enable_vram_management(vae_dtype=torch.float32)
    print("Model loaded.")

    # ------------------------------------------------------------------
    # 4. Sliding-window inference with temporal blending
    # ------------------------------------------------------------------
    device = next(pipe.vace.parameters()).device
    ws = args.window_size
    border_width = ws - args.window_stride

    # Streaming buffers: fixed size = window_size
    value_buf = torch.zeros(
        (ws, args.height, args.width, 3), device=device, dtype=torch.float32
    )
    weight_buf = torch.zeros((ws,), device=device, dtype=torch.float32)
    buf_len = 0
    first_unwritten = 0  # next global frame index to write (0-based)

    # Reference image for VACE conditioning.
    # When --use_prev_window_reference, updated from the previous window's
    # output at index window_stride (time-aligned with next window's start).
    current_reference = None  # will be set from first frame if needed

    for t in range(0, T_total, args.window_stride):
        # Skip windows that are fully covered by the previous window
        # (i.e., the previous window already reached the end of the video).
        if (
            t - args.window_stride >= 0
            and t - args.window_stride + ws >= T_total
        ):
            continue

        t_end = min(t + ws, T_total)
        window_files = frame_names[t:t_end]
        L_real = len(window_files)

        # Pad short final window by repeating the last frame so the model
        # always runs at its trained window_size (quality degrades otherwise).
        L_padded = L_real
        if L_real < ws:
            pad_count = ws - L_real
            window_files = window_files + [window_files[-1]] * pad_count
            L_padded = ws

        print(
            f"  Window [{t}:{t_end}] ({L_real} frames"
            + (f", padded to {L_padded}" if L_padded != L_real else "")
            + f"), global frames written so far: {first_unwritten}"
        )

        # --- Load and preprocess this window ---
        vace_video, vace_mask, all_masks = load_video_slice(
            image_folder=args.input_path,
            file_list=window_files,
            height=args.height,
            width=args.width,
            crop_and_resize=args.crop_and_resize,
            srgb_to_lg=args.srgb_to_lg,
            add_mask=args.add_mask,
            use_under_exposure_mask=args.use_under_exposure_mask,
            over_luma_thr_srgb=args.over_luma_thr_srgb,
            under_luma_thr_srgb=args.under_luma_thr_srgb,
            colour_utils=colour_utils,
        )

        vace_video_mask = None
        if args.add_mask:
            vace_video_mask = vace_mask

        # --- Run pipeline ---
        video_chunk = pipe(
            height=args.height,
            width=args.width,
            prompt=args.prompt,
            negative_prompt=NEGATIVE_PROMPT,
            seed=args.seed,
            tiled=False,
            is_hdr=True,
            num_inference_steps=args.num_inference_steps,
            optimize_mean_std=False,
            cfg_scale=args.cfg_scale,
            num_frames=L_padded,
            vace_video=vace_video,
            srgb_to_lg=args.srgb_to_lg,
            vace_video_mask=vace_video_mask,
            vace_reference_image=current_reference,
        )

        # video_chunk is a list of [H, W, C] tensors in Log space
        video_chunk = torch.stack(video_chunk)               # [L_padded, H, W, C]
        # Discard padded frames — keep only the real ones
        L = L_real
        video_chunk = video_chunk[:L]
        video_chunk = torch.clamp_min(video_chunk, 0.0)
        video_chunk = video_chunk.permute(0, 3, 1, 2).contiguous()  # [L, C, H, W] log

        # --- Update reference for next window ---
        if args.use_prev_window_reference:
            if L > args.window_stride:
                ref_idx = args.window_stride
            else:
                ref_idx = L - 1
            current_reference = (
                video_chunk[ref_idx]
                .clone()
                .detach()
                .to(device=device, dtype=torch.float32)
            )  # [C, H, W] in log space

        # --- Convert Log -> Linear for EXR output ---
        video_chunk = Log_to_Lin(video_chunk)
        video_chunk = video_chunk.permute(0, 2, 3, 1).contiguous()  # [L, H, W, C]

        # --- Temporal blend mask ---
        is_first_window = t == 0
        is_last_window = t_end == T_total
        mask_1d = build_1d_mask(
            L, is_first_window, is_last_window, border_width,
            device=device, dtype=torch.float32,
        )  # [L]
        mask_4d = mask_1d.view(L, 1, 1, 1)   # broadcastable over [L, H, W, C]
        chunk_weight = mask_1d                 # [L]

        # --- Accumulate into streaming buffer ---
        if is_first_window:
            value_buf[0:L] = video_chunk * mask_4d
            weight_buf[0:L] = chunk_weight
            buf_len = L
        else:
            # The first ``overlap_len`` frames in the buffer overlap with
            # the first ``overlap_len`` frames of this chunk.
            overlap_len = min(buf_len, L)
            value_buf[0:overlap_len] += (
                video_chunk[0:overlap_len] * mask_4d[0:overlap_len]
            )
            weight_buf[0:overlap_len] += chunk_weight[0:overlap_len]
            # Any new frames beyond the current buffer length
            if L > overlap_len:
                value_buf[overlap_len:L] = (
                    video_chunk[overlap_len:L] * mask_4d[overlap_len:L]
                )
                weight_buf[overlap_len:L] = chunk_weight[overlap_len:L]
            buf_len = L

        # --- Write the oldest window_stride frames (they are now complete) ---
        write_count = min(args.window_stride, buf_len)
        out_slice = value_buf[0:write_count] / (
            weight_buf[0:write_count].view(write_count, 1, 1, 1).clamp(min=1e-8)
        )
        for i in range(write_count):
            frame_idx_1based = first_unwritten + i + 1
            pyexr.write(
                os.path.join(args.output_dir, f"frame_{frame_idx_1based:04d}.exr"),
                out_slice[i].detach().cpu().numpy(),
            )
        first_unwritten += write_count

        # --- Shift buffer left by write_count ---
        remaining = buf_len - write_count
        if remaining > 0:
            value_buf[0:remaining] = value_buf[write_count:buf_len].clone()
            weight_buf[0:remaining] = weight_buf[write_count:buf_len].clone()
        buf_len = remaining

    # ------------------------------------------------------------------
    # 5. Flush remaining frames in buffer (tail of last window)
    # ------------------------------------------------------------------
    if buf_len > 0:
        out_slice = value_buf[0:buf_len] / (
            weight_buf[0:buf_len].view(buf_len, 1, 1, 1).clamp(min=1e-8)
        )
        for i in range(buf_len):
            frame_idx_1based = first_unwritten + i + 1
            pyexr.write(
                os.path.join(args.output_dir, f"frame_{frame_idx_1based:04d}.exr"),
                out_slice[i].detach().cpu().numpy(),
            )
        first_unwritten += buf_len

    torch.cuda.empty_cache()
    print(f"Done. Wrote {first_unwritten} EXR frames to {args.output_dir}")


if __name__ == "__main__":
    main()
