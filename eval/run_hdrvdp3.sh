#!/usr/bin/env bash
set -euo pipefail

MATLAB_BIN="${MATLAB_BIN:-matlab}"
EVAL_ROOT="${EVAL_ROOT:-./HDRVDP3_Eval}"
REF_DIR_DEFAULT="${REF_DIR:-./SI_HDR_Dataset/reference}"

usage() {
  cat <<'EOF'
Usage:
  run_hdrvdp3.sh -g GEN_DIR -m SCALE_MODE -o OUT_CSV [options]

Required:
  -g, --gen-dir        Generated frames folder
  -m, --scale-mode     max | percentile
  -o, --out-csv        Output CSV path

Optional:
  -r, --ref-dir        Reference folder (default: ./SI_HDR_Dataset/reference)
  --ppd N              Pixels-per-degree (default: 60)
  --lpeak N            Peak luminance cd/m^2 (default: 1000)
  --scalep P           Percentile for 'percentile' mode (default: 99.9)
  --quiet true|false   Pass Quiet to MATLAB (default: true)

  --ours               Enable ours folder matching (recursive **/frame_00.exr, stem=parent dir)
  --ours-frame NAME    Frame filename to pick in ours mode (default: frame_00.exr)

Env overrides:
  MATLAB_BIN           Path to matlab binary
  EVAL_ROOT            Path to HDRVDP3_Eval root (added via addpath(genpath))

Examples:
  # flat gen folder
  ./run_hdrvdp3.sh -g /path/to/clip_95 -m max -o hdrvdp3_eval_results/clip95_max.csv

  # ours mode (nested .../<stem>/frame_00.exr)
  ./run_hdrvdp3.sh --ours -g /path/to/results/clip_95 \
    -m max -o hdrvdp3_eval_results/clip95_ours_max.csv

  # ours mode but choose another frame
  ./run_hdrvdp3.sh --ours --ours-frame frame_05.exr -g /.../no_under_mask -m percentile -o out.csv --scalep 99.9
EOF
}

# defaults
GEN_DIR=""
REF_DIR="$REF_DIR_DEFAULT"
SCALE_MODE=""
OUT_CSV=""
PPD=60
LPEAK=1000
SCALEP=99.9
QUIET=true

OURS=false
OURS_FRAME="frame_00.exr"

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gen-dir) GEN_DIR="$2"; shift 2;;
    -r|--ref-dir) REF_DIR="$2"; shift 2;;
    -m|--scale-mode) SCALE_MODE="$2"; shift 2;;
    -o|--out-csv) OUT_CSV="$2"; shift 2;;
    --ppd) PPD="$2"; shift 2;;
    --lpeak) LPEAK="$2"; shift 2;;
    --scalep) SCALEP="$2"; shift 2;;
    --quiet) QUIET="$2"; shift 2;;

    --ours) OURS=true; shift 1;;
    --ours-frame) OURS_FRAME="$2"; shift 2;;

    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$GEN_DIR" || -z "$SCALE_MODE" || -z "$OUT_CSV" ]]; then
  echo "Error: -g/--gen-dir, -m/--scale-mode, -o/--out-csv are required."
  usage
  exit 1
fi

if [[ "$SCALE_MODE" != "max" && "$SCALE_MODE" != "percentile" ]]; then
  echo "Error: --scale-mode must be 'max' or 'percentile' (got: $SCALE_MODE)"
  exit 1
fi

# ensure output dir exists
OUT_DIR="$(dirname "$OUT_CSV")"
mkdir -p "$OUT_DIR"

# Normalize paths (optional; useful when caller passes relative paths)
GEN_DIR="$(readlink -f "$GEN_DIR" 2>/dev/null || echo "$GEN_DIR")"
REF_DIR="$(readlink -f "$REF_DIR" 2>/dev/null || echo "$REF_DIR")"
EVAL_ROOT="$(readlink -f "$EVAL_ROOT" 2>/dev/null || echo "$EVAL_ROOT")"
OUT_CSV="$(readlink -m "$OUT_CSV" 2>/dev/null || echo "$OUT_CSV")"

# Build MATLAB command pieces
MATLAB_PREFIX="addpath(genpath('${EVAL_ROOT}'));"
MATLAB_ARGS="'OutCSV','${OUT_CSV}','PPD',${PPD},'LPeak',${LPEAK},'Quiet',${QUIET}"

if [[ "$OURS" == "true" ]]; then
  MATLAB_ARGS="${MATLAB_ARGS},'Ours',true,'OursFrameName','${OURS_FRAME}'"
fi

if [[ "$SCALE_MODE" == "percentile" ]]; then
  MATLAB_ARGS="${MATLAB_ARGS},'ScaleMode','percentile','ScaleP',${SCALEP}"
else
  MATLAB_ARGS="${MATLAB_ARGS},'ScaleMode','max'"
fi

MATLAB_CMD="${MATLAB_PREFIX} run_hdrvdp3_dir('${GEN_DIR}','${REF_DIR}',${MATLAB_ARGS});"

echo "[run_hdrvdp3] MATLAB_BIN:  $MATLAB_BIN"
echo "[run_hdrvdp3] EVAL_ROOT:   $EVAL_ROOT"
echo "[run_hdrvdp3] GEN_DIR:     $GEN_DIR"
echo "[run_hdrvdp3] REF_DIR:     $REF_DIR"
echo "[run_hdrvdp3] OUT_CSV:     $OUT_CSV"
echo "[run_hdrvdp3] PPD:         $PPD"
echo "[run_hdrvdp3] LPEAK:       $LPEAK"
echo "[run_hdrvdp3] SCALE_MODE:  $SCALE_MODE"
if [[ "$SCALE_MODE" == "percentile" ]]; then
  echo "[run_hdrvdp3] SCALEP:      $SCALEP"
fi
echo "[run_hdrvdp3] OURS:        $OURS"
if [[ "$OURS" == "true" ]]; then
  echo "[run_hdrvdp3] OURS_FRAME:  $OURS_FRAME"
fi

"$MATLAB_BIN" -batch "$MATLAB_CMD"
