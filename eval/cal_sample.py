#!/usr/bin/env python3
"""
Sample evaluation script — compute all metrics for a single gen/gt folder pair.

Gen layout:  gen_dir/{frame}.exr
GT layout:   gt_dir/{frame}.exr   (optional; enables FovVideoVDP)

Metrics:
  NR:  MUSIQ, CLIPIQA, PU21-PIQE, DOVER (optional)
  FR:  FovVideoVDP (requires gt_dir)

Usage:
    python cal_sample.py \
        --gen_dir /path/to/generated_exr_frames \
        --gt_dir  /path/to/ground_truth_exr_frames \
        --out_csv results/sample_eval.csv

    # NR-only (no ground truth):
    python cal_sample.py \
        --gen_dir /path/to/generated_exr_frames \
        --out_csv results/sample_eval.csv
"""

import argparse
import csv
import os
import random
import tempfile
from pathlib import Path

import numpy as np

from cal_metrics import (
    init_pyiqa, compute_musiq_clipiqa,
    init_pyiqa_pu21, compute_pu21_piqe,
    compute_fovvdp,
    list_frames_gen, list_frames_gt, align_frames,
    encode_mp4_from_tonemapped, run_dover_on_mp4,
    mean_key, write_csv,
)


def main():
    ap = argparse.ArgumentParser(description="Evaluate HDR frames (gen vs gt)")
    ap.add_argument("--gen_dir", required=True, help="Folder of generated EXR frames")
    ap.add_argument("--gt_dir", default=None, help="Folder of ground-truth EXR frames (enables FovVideoVDP)")
    ap.add_argument("--out_csv", default="eval_results.csv", help="Output CSV path")

    ap.add_argument("--max_frames", type=int, default=None, help="Max frames to evaluate")
    ap.add_argument("--stride", type=int, default=1, help="Frame stride for subsampling")

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip_variant", default="clipiqa", choices=["clipiqa", "clipiqa+"])

    ap.add_argument("--fovvdp_display", default="standard_hdr_linear")
    ap.add_argument("--fovvdp_peak_nits", type=float, default=1000.0)

    ap.add_argument("--dover_repo", default=None, help="Path to DOVER repo (enables DOVER metric)")
    ap.add_argument("--dover_fps", type=int, default=24)
    ap.add_argument("--dover_device", default="cuda")

    args = ap.parse_args()

    import torch
    os.environ["PYTHONHASHSEED"] = "34"
    random.seed(34)
    np.random.seed(34)
    torch.manual_seed(34)
    torch.cuda.manual_seed_all(34)

    gen_dir = Path(args.gen_dir)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    dover_repo = Path(args.dover_repo) if args.dover_repo else None

    if not gen_dir.exists():
        raise FileNotFoundError(f"gen_dir not found: {gen_dir}")

    gen_frames = list_frames_gen(gen_dir)
    if not gen_frames:
        raise RuntimeError(f"No EXR/HDR frames in {gen_dir}")
    print(f"Found {len(gen_frames)} generated frames in {gen_dir}")

    # ── Init metrics ──
    iqa = init_pyiqa(device=args.device, clip_variant=args.clip_variant)
    pu = init_pyiqa_pu21(device=args.device)

    row = {"gen_dir": str(gen_dir), "gt_dir": str(gt_dir) if gt_dir else ""}

    # ── NR: MUSIQ / CLIPIQA ──
    nr = compute_musiq_clipiqa(gen_frames, iqa, max_frames=args.max_frames, stride=args.stride)
    row.update(nr)
    print(f"MUSIQ={nr['musiq']:.4f}  CLIPIQA={nr['clipiqa']:.4f}  (frames={int(nr['n_frames_iqa'])})")

    # ── NR: PU21-PIQE ──
    pu_nr = compute_pu21_piqe(gen_frames, pu, max_frames=args.max_frames, stride=args.stride)
    row.update(pu_nr)
    print(f"PU21-PIQE={pu_nr['pu21_piqe']:.4f}  (k={pu_nr['pu21_scale_k']:.3g})")

    # ── FR: FovVideoVDP ──
    if gt_dir is not None and gt_dir.exists():
        gt_frames = list_frames_gt(gt_dir)
        if gt_frames:
            gen_a, gt_a = align_frames(gen_frames, gt_frames)
            print(f"Aligned {len(gen_a)} frame pairs for FovVideoVDP")
            fr = compute_fovvdp(
                gen_a, gt_a,
                display_name=args.fovvdp_display,
                peak_nits=args.fovvdp_peak_nits,
                max_frames=args.max_frames,
                stride=args.stride,
            )
            row.update(fr)
            print(f"FovVideoVDP(JOD)={fr['fovvdp_jod']:.4f}  (frames={int(fr['n_frames_fovvdp'])})")
        else:
            print(f"[WARN] No GT frames in {gt_dir}, skipping FovVideoVDP")
    elif gt_dir is not None:
        print(f"[WARN] gt_dir not found: {gt_dir}, skipping FovVideoVDP")

    # ── NR: DOVER ──
    if dover_repo is not None:
        with tempfile.TemporaryDirectory() as td:
            mp4_path = Path(td) / "eval_video.mp4"
            encode_mp4_from_tonemapped(
                gen_frames, mp4_path,
                fps=args.dover_fps,
                max_frames=args.max_frames,
                stride=args.stride,
            )
            dv = run_dover_on_mp4(
                dover_repo=dover_repo,
                mp4_path=mp4_path,
                device=args.dover_device,
                fusion=True,
            )
            row.update(dv)
            print(f"DOVER_fused={dv.get('dover_fused', float('nan')):.4f}")

    # ── Write CSV ──
    header = [
        "gen_dir", "gt_dir",
        "musiq", "clipiqa", "n_frames_iqa",
        "pu21_piqe", "n_frames_pu21", "pu21_scale_k",
        "fovvdp_jod", "n_frames_fovvdp", "fovvdp_display",
        "dover_fused", "dover_mode",
    ]
    write_csv([row], header, Path(args.out_csv))
    print(f"\nDone. Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
