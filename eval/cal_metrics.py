#!/usr/bin/env python3
"""
Metric functions for HDR evaluation.

Metrics:
  - MUSIQ, CLIPIQA (NR image quality on tonemapped LDR)
  - PU21-PIQE (NR quality on PU21-encoded HDR luminance)
  - FovVideoVDP (FR HDR visual difference)
  - DOVER (NR video quality on tonemapped MP4)
  - FID (distribution distance on tonemapped LDR patches)
"""

import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────── IO ─────────────────────────────────────

def read_hdr_image(path: str) -> np.ndarray:
    """Load .exr/.hdr -> float32 RGB HxWx3 in linear space."""
    try:
        import OpenImageIO as oiio
    except Exception as e:
        raise RuntimeError(
            "OpenImageIO is required. Install: pip install OpenImageIO "
            "or conda install -c conda-forge openimageio\n"
            f"Import error: {e}"
        )

    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise RuntimeError(f"Failed to open image: {path}")
    spec = inp.spec()
    w, h, c = spec.width, spec.height, spec.nchannels
    arr = inp.read_image(format=oiio.FLOAT)
    inp.close()

    arr = np.array(arr, dtype=np.float32).reshape(h, w, c)
    if c >= 3:
        rgb = arr[..., :3]
    elif c == 1:
        rgb = np.repeat(arr, 3, axis=2)
    else:
        raise RuntimeError(f"Unsupported channel count ({c}) in {path}")

    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = np.clip(rgb, 0.0, None)
    return rgb


# ──────────────────────────── Color ──────────────────────────────────

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    thr = 0.0031308
    srgb = np.where(x <= thr, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a)
    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


def reinhard_tonemap(rgb_lin: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Luminance-based Reinhard: Y/(1+Y) with sRGB output."""
    rgb = np.clip(rgb_lin, 0.0, None)
    Y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    Y_percentile_max = np.percentile(Y, 99.9)
    if Y_percentile_max < 1.0:
        scale = 1.0 / Y_percentile_max
        rgb_tm = rgb * scale
    else:
        Y_tm = Y / (1.0 + Y)
        scale = Y_tm / (Y + eps)
        rgb_tm = rgb * scale[..., None]

    rgb_tm = np.clip(rgb_tm, 0.0, 1.0)
    return linear_to_srgb(rgb_tm)


def reinhard_tonemap_opencv(
    rgb_lin: np.ndarray,
    gamma: float = 2.4,
    intensity: float = 0.0,
    light_adapt: float = 1.0,
    color_adapt: float = 0.0,
) -> np.ndarray:
    """OpenCV Reinhard tone mapping. Output: HxWx3 float32 [0,1]."""
    import cv2

    rgb = np.nan_to_num(rgb_lin, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    rgb = np.clip(rgb, 1e-6, None)
    bgr = rgb[..., ::-1]
    tonemap = cv2.createTonemapReinhard(
        gamma=gamma, intensity=intensity,
        light_adapt=light_adapt, color_adapt=color_adapt,
    )
    ldr_bgr = tonemap.process(bgr)
    ldr_bgr = np.clip(ldr_bgr, 0.0, 1.0).astype(np.float32)
    return ldr_bgr[..., ::-1]


# ──────────────────────────── Frame listing ──────────────────────────

def list_frames_gen(video_dir: Path) -> List[Path]:
    exts = {".exr", ".hdr"}
    frames = [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    frames.sort(key=lambda p: p.name)
    return frames


def list_frames_gt(gt_video_hdr_dir: Path) -> List[Path]:
    frames = [p for p in gt_video_hdr_dir.iterdir() if p.is_file() and p.suffix.lower() == ".exr"]
    frames.sort(key=lambda p: p.name)
    return frames


def align_frames(gen_frames: List[Path], gt_frames: List[Path]) -> Tuple[List[Path], List[Path]]:
    """Try filename-based alignment; fallback to index-based truncation."""
    if not gt_frames:
        return gen_frames, gt_frames

    gt_map = {p.name: p for p in gt_frames}
    paired = [(g, gt_map[g.name]) for g in gen_frames if g.name in gt_map]
    if len(paired) >= max(1, int(0.7 * min(len(gen_frames), len(gt_frames)))):
        return [a for a, _ in paired], [b for _, b in paired]

    n = min(len(gen_frames), len(gt_frames))
    return gen_frames[:n], gt_frames[:n]


def np_to_torch_img01(img01: np.ndarray, device: str):
    import torch
    if img01.ndim != 3 or img01.shape[2] != 3:
        raise ValueError(f"Expected HxWx3, got {img01.shape}")
    t = torch.from_numpy(img01).permute(2, 0, 1).unsqueeze(0).contiguous()
    t = t.to(device=device, dtype=torch.float32)
    return t


def _subsample_indices(total: int, max_frames: Optional[int] = None, stride: int = 1) -> List[int]:
    idxs = list(range(0, total, stride))
    if max_frames is not None:
        idxs = idxs[:max_frames]
    return idxs


# ──────────────────────────── MUSIQ / CLIPIQA ────────────────────────

@dataclass
class IQAMetrics:
    musiq: Optional[object] = None
    clipiqa: Optional[object] = None
    device: str = "cuda"


def init_pyiqa(device: str, clip_variant: str = "clipiqa") -> IQAMetrics:
    import torch
    import pyiqa
    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    musiq_metric = pyiqa.create_metric("musiq", device=dev)
    clip_metric = pyiqa.create_metric(clip_variant, device=dev)
    return IQAMetrics(musiq=musiq_metric, clipiqa=clip_metric, device=str(dev))


def compute_musiq_clipiqa(
    frames_hdr: List[Path],
    iqa: IQAMetrics,
    max_frames: Optional[int] = None,
    stride: int = 1,
) -> Dict[str, float]:
    """HDR frames -> Reinhard tonemap -> MUSIQ/CLIPIQA per frame -> average."""
    import torch

    idxs = _subsample_indices(len(frames_hdr), max_frames, stride)
    musiq_scores, clip_scores = [], []

    with torch.no_grad():
        for i in idxs:
            rgb = read_hdr_image(str(frames_hdr[i]))
            ldr = reinhard_tonemap(rgb)
            t = np_to_torch_img01(ldr, iqa.device)
            musiq_scores.append(float(iqa.musiq(t).item()))
            clip_scores.append(float(iqa.clipiqa(t).item()))

    return {
        "musiq": float(np.mean(musiq_scores)) if musiq_scores else float("nan"),
        "clipiqa": float(np.mean(clip_scores)) if clip_scores else float("nan"),
        "n_frames_iqa": float(len(idxs)),
    }


# ──────────────────────────── PU21-PIQE ──────────────────────────────

class PU21Encoder:
    """PU21 encoder (Mantiuk & Azimi, PCS 2021). Input Y in cd/m^2."""

    def __init__(self, enc_type: str = "banding_glare"):
        self.L_min = 0.005
        self.L_max = 10000.0
        params = {
            "banding":       [1.070275272, 0.4088273932, 0.153224308,   0.2520326168, 1.063512885,  1.14115047,  521.4527484],
            "banding_glare": [0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627, 0.09150303166, 0.9099517204, 596.3148142],
            "peaks":         [1.043882782, 0.6459495343, 0.3194584211,  0.374025247,  1.114783422,  1.095360363, 384.9217577],
            "peaks_glare":   [816.885024,  1479.463946,  0.001253215609, 0.9329636822, 0.06746643971, 1.573435413, 419.6006374],
        }
        if enc_type not in params:
            raise ValueError(f"Unknown PU21 type: {enc_type}")
        self.par = params[enc_type]

    def encode(self, Y: np.ndarray) -> np.ndarray:
        p = self.par
        Y = np.clip(Y, self.L_min, self.L_max)
        V = p[6] * (((p[0] + p[1] * (Y ** p[3])) / (1.0 + p[2] * (Y ** p[3]))) ** p[4] - p[5])
        return np.maximum(V, 0.0).astype(np.float32)


@dataclass
class PU21Metrics:
    piqe: Optional[object] = None
    device: str = "cuda"
    enc_type: str = "banding_glare"
    peak_nits: float = 1000.0
    perc: float = 99.9


def init_pyiqa_pu21(device: str) -> PU21Metrics:
    import torch
    import pyiqa
    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    piqe_metric = pyiqa.create_metric("piqe", device=dev)
    return PU21Metrics(piqe=piqe_metric, device=str(dev))


def _estimate_k_from_frames(
    frames_hdr: List[Path], peak_nits: float, perc: float,
    max_frames: Optional[int] = None, stride: int = 1,
) -> float:
    idxs = _subsample_indices(len(frames_hdr), max_frames, stride)
    if not idxs:
        return 1.0
    Ys = []
    for i in idxs:
        rgb = read_hdr_image(str(frames_hdr[i])).astype(np.float32)
        Y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        Ys.append(np.percentile(Y, perc))
    p = float(np.mean(Ys))
    return float(peak_nits / max(p, 1e-6))


def compute_pu21_piqe(
    frames_hdr: List[Path],
    pu: PU21Metrics,
    max_frames: Optional[int] = None,
    stride: int = 1,
) -> Dict[str, float]:
    """HDR frames -> PU21 luminance encode -> PIQE per frame -> average."""
    import torch

    idxs = _subsample_indices(len(frames_hdr), max_frames, stride)
    k = _estimate_k_from_frames(frames_hdr, pu.peak_nits, pu.perc, max_frames, stride)
    enc = PU21Encoder(pu.enc_type)

    piqe_scores = []
    with torch.no_grad():
        for i in idxs:
            rgb = read_hdr_image(str(frames_hdr[i])).astype(np.float32) * k
            Y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            V = enc.encode(Y)
            Vn = np.clip(V / 600.0, 0.0, 1.0).astype(np.float32)
            img = np.repeat(Vn[..., None], 3, axis=2)
            t = np_to_torch_img01(img, pu.device)
            piqe_scores.append(float(pu.piqe(t).item()))

    return {
        "pu21_piqe": float(np.mean(piqe_scores)) if piqe_scores else float("nan"),
        "n_frames_pu21": float(len(idxs)),
        "pu21_scale_k": float(k),
        "pu21_peak_nits": float(pu.peak_nits),
        "pu21_perc": float(pu.perc),
        "pu21_type": pu.enc_type,
    }


# ──────────────────────────── FovVideoVDP ────────────────────────────

def compute_fovvdp(
    gen_frames: List[Path],
    gt_frames: List[Path],
    display_name: str = "standard_hdr_linear",
    max_frames: Optional[int] = None,
    stride: int = 1,
    peak_nits: Optional[float] = None,
    perc: float = 99.9,
) -> Dict[str, float]:
    """Video-level FovVideoVDP with temporal model."""
    import pyfvvdp

    idxs = _subsample_indices(min(len(gen_frames), len(gt_frames)), max_frames, stride)
    gen_frames = [gen_frames[i] for i in idxs]
    gt_frames = [gt_frames[i] for i in idxs]

    k = None
    if peak_nits is not None:
        Ys = []
        for r in gt_frames:
            rgb = read_hdr_image(str(r))
            Y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            Ys.append(np.percentile(Y, perc))
        p = float(np.mean(Ys))
        k = peak_nits / max(p, 1e-6)

    tests, refs = [], []
    for g, r in zip(gen_frames, gt_frames):
        I_test = read_hdr_image(str(g))
        I_ref = read_hdr_image(str(r))
        if k is not None:
            I_test = I_test * k
            I_ref = I_ref * k
        tests.append(I_test.astype(np.float32))
        refs.append(I_ref.astype(np.float32))

    I_test_v = np.stack(tests, axis=0)
    I_ref_v = np.stack(refs, axis=0)

    if peak_nits is not None:
        disp_photo = pyfvvdp.fvvdp_display_photo_absolute(peak_nits)
        fv = pyfvvdp.fvvdp(display_name="standard_hdr_linear", display_photometry=disp_photo)
    else:
        fv = pyfvvdp.fvvdp(display_name=display_name)

    try:
        q, stats = fv.predict(I_test_v, I_ref_v, dim_order="THWC")
        return {
            "fovvdp_jod": float(q),
            "n_frames_fovvdp": float(I_test_v.shape[0]),
            "fovvdp_display": display_name,
            "fovvdp_mode": "video",
            "fovvdp_scale_k": float(k) if k is not None else "",
        }
    except Exception as e:
        q_scores = []
        for t in range(I_test_v.shape[0]):
            q_t, _ = fv.predict(I_test_v[t], I_ref_v[t], dim_order="HWC")
            q_scores.append(float(q_t))
        return {
            "fovvdp_jod": float(np.mean(q_scores)) if q_scores else float("nan"),
            "n_frames_fovvdp": float(len(q_scores)),
            "fovvdp_display": display_name,
            "fovvdp_mode": f"framewise_fallback({type(e).__name__})",
            "fovvdp_scale_k": float(k) if k is not None else "",
        }


# ──────────────────────────── DOVER ──────────────────────────────────

def encode_mp4_from_tonemapped(
    frames_hdr: List[Path],
    out_mp4: Path,
    fps: int = 24,
    max_frames: Optional[int] = None,
    stride: int = 1,
):
    """Tonemap HDR frames -> write MP4 via imageio."""
    import imageio

    idxs = _subsample_indices(len(frames_hdr), max_frames, stride)
    if not idxs:
        raise RuntimeError("No frames to encode.")

    writer = imageio.get_writer(
        str(out_mp4), fps=fps, codec="libx264",
        pixelformat="yuv420p", quality=8,
    )
    try:
        for i in idxs:
            rgb = read_hdr_image(str(frames_hdr[i]))
            ldr = reinhard_tonemap(rgb)
            frame = (ldr * 255.0 + 0.5).astype(np.uint8)
            writer.append_data(frame)
    finally:
        writer.close()


def run_dover_on_mp4(
    dover_repo: Path,
    mp4_path: Path,
    device: str = "cuda",
    opt_path: Optional[Path] = None,
    fusion: bool = True,
) -> dict:
    dover_repo = Path(dover_repo)
    eval_py = dover_repo / "evaluate_one_video.py"
    if not eval_py.exists():
        cand = list(dover_repo.rglob("evaluate_one_video.py"))
        if cand:
            eval_py = cand[0]
        else:
            raise RuntimeError(f"Cannot find evaluate_one_video.py under {dover_repo}")

    if opt_path is None:
        opt_path = dover_repo / "dover.yml"
    else:
        opt_path = Path(opt_path)

    cmd = ["python", str(eval_py), "-v", str(mp4_path), "-d", device, "-o", str(opt_path)]
    if fusion:
        cmd.append("-f")

    p = subprocess.run(cmd, cwd=str(dover_repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"DOVER failed (code={p.returncode})\nCMD: {' '.join(cmd)}\n"
            f"STDERR:\n{p.stderr}\nSTDOUT:\n{p.stdout}\n"
        )

    txt = p.stdout + "\n" + p.stderr
    out = {}
    if fusion:
        m = re.search(r"Normalized fused overall score.*?:\s*([0-9]*\.?[0-9]+)", txt)
        out["dover_fused"] = float(m.group(1)) if m else float("nan")
        out["dover_mode"] = "fusion"
    else:
        out["dover_mode"] = "disentangled"
        out["dover_tqe"] = float("nan")
        out["dover_aqe"] = float("nan")
    return out


# ──────────────────────────── FID ────────────────────────────────────

def random_crops(img01: np.ndarray, crop: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """img01: HxWx3 float32 [0,1] -> n x 3 x crop x crop."""
    H, W, _ = img01.shape
    if H < crop or W < crop:
        import torch
        import torch.nn.functional as F
        t = torch.from_numpy(img01).permute(2, 0, 1).unsqueeze(0)
        newH, newW = max(H, crop), max(W, crop)
        t = F.interpolate(t, size=(newH, newW), mode="bilinear", align_corners=False)
        img01 = t[0].permute(1, 2, 0).numpy().astype(np.float32)
        H, W, _ = img01.shape

    ys = rng.integers(0, H - crop + 1, size=n)
    xs = rng.integers(0, W - crop + 1, size=n)
    out = np.empty((n, 3, crop, crop), dtype=np.float32)
    for i, (y, x) in enumerate(zip(ys, xs)):
        p = img01[y:y + crop, x:x + crop, :]
        out[i] = np.transpose(p, (2, 0, 1))
    return out


def save_patches_png(patches_chw: np.ndarray, out_dir: Path, prefix: str, start_idx: int = 0) -> int:
    import imageio.v2 as imageio
    out_dir.mkdir(parents=True, exist_ok=True)
    n = patches_chw.shape[0]
    for i in range(n):
        img = np.transpose(patches_chw[i], (1, 2, 0))
        img_u8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        imageio.imwrite(out_dir / f"{prefix}_{start_idx + i:06d}.png", img_u8)
    return n


def compute_fid(gen_folder, gt_folder, batch_size=64, device=None, dims=2048, num_workers=4):
    """FID via pytorch-fid on two folders of PNG patches."""
    from pytorch_fid import fid_score
    fid = fid_score.calculate_fid_given_paths(
        [str(gt_folder), str(gen_folder)],
        batch_size=batch_size,
        device=device or "cuda",
        dims=dims,
        num_workers=num_workers,
    )
    return float(fid)


# ──────────────────────────── CSV helpers ────────────────────────────

def mean_key(rows: List[dict], key: str):
    vals = []
    for r in rows:
        v = r.get(key, None)
        if v is None or v == "" or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            continue
        try:
            vals.append(float(v))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else ""


def write_csv(rows: List[dict], header: List[str], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})
