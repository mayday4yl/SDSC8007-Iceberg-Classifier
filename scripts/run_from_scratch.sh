#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-artifacts/from_scratch_${RUN_ID}}"
PRED_DIR="${RUN_ROOT}/predictions"
MODEL_DIR="${RUN_ROOT}/models"
REPORT_DIR="${RUN_ROOT}/reports"
FILM_PRETRAINED_ARGS=()
if [[ "${NO_PRETRAINED:-0}" == "1" ]]; then
  FILM_PRETRAINED_ARGS+=(--no-pretrained)
fi

mkdir -p "$PRED_DIR" "$MODEL_DIR" "$REPORT_DIR"
exec > >(tee -a "${RUN_ROOT}/run.log") 2>&1

echo "Project root: $ROOT"
echo "Run root:     $RUN_ROOT"
echo "Python:       $($PYTHON_BIN -c 'import sys; print(sys.executable)')"

for required in data/processed/train.json data/processed/test.json data/processed/sample_submission.csv; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required data file: $required" >&2
    echo "Put the Kaggle-extracted raw files into data/processed/ before running this script." >&2
    exit 1
  fi
done

latest_oof() {
  local pattern="$1"
  ls -t "${PRED_DIR}"/oof_${pattern}_*.csv | head -1
}

matching_submission() {
  local oof_path="$1"
  local name
  name="$(basename "$oof_path")"
  echo "${PRED_DIR}/submission_${name#oof_}"
}

run_train() {
  "$PYTHON_BIN" -u scripts/train_cnn.py "$@" \
    --output-dir "$PRED_DIR" \
    --model-dir "$MODEL_DIR" \
    --report-dir "$REPORT_DIR"
}

echo
echo "[1/7] Build cache from raw Kaggle JSON"
"$PYTHON_BIN" scripts/build_cache.py

echo
echo "[2/7] Base model A: compact CNN / ResNet-style block, dB image, angle auxiliary"
run_train \
  --arch resnet \
  --folds 5 \
  --epochs 120 \
  --patience 25 \
  --batch-size 64 \
  --width 48 \
  --dropout 0.20 \
  --lr 0.001 \
  --image-mode db \
  --aux-features angle \
  --tta
BASE_A_OOF="$(latest_oof "resnet_db_stratified_seed2026")"
BASE_A_SUB="$(matching_submission "$BASE_A_OOF")"
echo "Base A CV prediction file: $BASE_A_OOF"

echo
echo "[3/7] Base model B: VGG-style CNN, dB image, angle auxiliary"
run_train \
  --arch vgg \
  --folds 5 \
  --epochs 120 \
  --patience 25 \
  --batch-size 64 \
  --width 48 \
  --dropout 0.30 \
  --lr 0.001 \
  --image-mode db \
  --aux-features angle \
  --tta
BASE_B_OOF="$(latest_oof "vgg_db_stratified_seed2026")"
BASE_B_SUB="$(matching_submission "$BASE_B_OOF")"
echo "Base B CV prediction file: $BASE_B_OOF"

echo
echo "[4/7] Base model C: pseudo-label filtered CNN using Base A submission"
run_train \
  --arch resnet \
  --folds 5 \
  --epochs 120 \
  --patience 25 \
  --batch-size 64 \
  --width 48 \
  --dropout 0.20 \
  --lr 0.001 \
  --image-mode db \
  --aux-features angle \
  --pseudo-sub "$BASE_A_SUB" \
  --pseudo-low 0.03 \
  --pseudo-high 0.97 \
  --pseudo-weight 0.35 \
  --tta
BASE_C_OOF="$(latest_oof "resnet_db_stratified_seed2026")"
BASE_C_SUB="$(matching_submission "$BASE_C_OOF")"
echo "Base C CV prediction file: $BASE_C_OOF"

echo
echo "[5/7] Base model D: pretrained FiLM ResNet34, dB image, angle modulation"
run_train \
  --arch film_resnet \
  --folds 5 \
  --epochs 80 \
  --patience 15 \
  --batch-size 32 \
  --dropout 0.30 \
  --lr 0.0003 \
  --image-mode db \
  --aux-features angle \
  "${FILM_PRETRAINED_ARGS[@]}" \
  --tta
BASE_D_OOF="$(latest_oof "film_resnet_db_stratified_seed2026")"
BASE_D_SUB="$(matching_submission "$BASE_D_OOF")"
echo "Base D CV prediction file: $BASE_D_OOF"

OOF_BASE=("$BASE_A_OOF" "$BASE_B_OOF" "$BASE_C_OOF" "$BASE_D_OOF")
SUB_BASE=("$BASE_A_SUB" "$BASE_B_SUB" "$BASE_C_SUB" "$BASE_D_SUB")

echo
echo "[6/7] Strict Layer-2 angle-aware stacking bag"
for seed in 2026 2027 2028 2029; do
  "$PYTHON_BIN" -u scripts/angle_stack.py \
    --oof "${OOF_BASE[@]}" \
    --sub "${SUB_BASE[@]}" \
    --model lgbm \
    --seed "$seed" \
    --output-dir "$PRED_DIR" \
    --report-dir "$REPORT_DIR"

  "$PYTHON_BIN" -u scripts/angle_stack.py \
    --oof "${OOF_BASE[@]}" \
    --sub "${SUB_BASE[@]}" \
    --model logreg \
    --seed "$seed" \
    --output-dir "$PRED_DIR" \
    --report-dir "$REPORT_DIR"
done

echo
echo "[7/7] Final strict blend"
OOF_STRICT=()
SUB_STRICT=()
for p in "${PRED_DIR}"/oof_strict_angle_stack_lgbm_4models_*.csv "${PRED_DIR}"/oof_strict_angle_stack_logreg_4models_*.csv; do
  [[ -e "$p" ]] || continue
  suffix="${p#${PRED_DIR}/oof_strict_}"
  OOF_STRICT+=("$p")
  SUB_STRICT+=("${PRED_DIR}/submission_${suffix}")
done

"$PYTHON_BIN" scripts/blend_predictions.py \
  --oof "${OOF_STRICT[@]}" \
  --sub "${SUB_STRICT[@]}" \
  --output-dir "$PRED_DIR" \
  --report-dir "$REPORT_DIR"
STRICT_BLEND_OOF="$(ls -t "${PRED_DIR}"/oof_blend_*models_*.csv | head -1)"
STRICT_PREFIX="$(basename "$STRICT_BLEND_OOF")"
STRICT_PREFIX="${STRICT_PREFIX#oof_}"
STRICT_PREFIX="${STRICT_PREFIX%.csv}"
FINAL_BLEND_OOF="$STRICT_BLEND_OOF"
FINAL_PREFIX="$STRICT_PREFIX"
FINAL_METRICS="${REPORT_DIR}/metrics_${FINAL_PREFIX}.json"
FINAL_SUB="${PRED_DIR}/submission_${FINAL_PREFIX}.csv"

echo
echo "Final CV prediction file: $FINAL_BLEND_OOF"
echo "Final submission:         $FINAL_SUB"
echo "Final metrics:            $FINAL_METRICS"
echo
"$PYTHON_BIN" scripts/verify_final_oof.py --oof "$FINAL_BLEND_OOF" --metrics "$FINAL_METRICS"

echo
echo "Done. All outputs for this from-scratch run are under: $RUN_ROOT"
