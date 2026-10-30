#!/usr/bin/env bash
# DianDog 楼梯 + 高度扫描（239D）：50% 平地全向 + 50% 楼梯前进，最高 20cm。
# 默认全新训练（不能续 flat 52D）。续跑同实验：RESUME=1
#   GUI=1 NUM_ENVS=4096 bash scripts/train_diandog_stairs_omni.sh
#   RESUME=1 GUI=0 NUM_ENVS=4096 bash scripts/train_diandog_stairs_omni.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WDOG="/home/hi/liu/wheeled_legged_dog"
ILAB="$WDOG/IsaacLab"

EXP_DIR="$ROOT/logs/rsl_rl/diandog_stairs_omni"
mkdir -p "$EXP_DIR"

source /home/hi/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab51
export ZSH_VERSION=""
export OMNI_KIT_ACCEPT_EULA=YES
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/home/hi/.Xauthority}"
export PYTHONUNBUFFERED=1
set +u
source "$ILAB/_isaac_sim/setup_conda_env.sh"
set -euo pipefail

export PYTHONPATH="\
${ROOT}/source/isaaclab_tasks:\
${ROOT}/source/isaaclab_assets:\
${ROOT}/source/isaaclab_rl:\
${ROOT}/source/isaaclab:\
${PYTHONPATH:-}"

cd "$ROOT"
TASK="${TASK:-DianDog-Stairs-Omni-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
GUI="${GUI:-1}"

CMD=(
  python -u scripts/reinforcement_learning/rsl_rl/train.py
  --task="$TASK"
  --num_envs="$NUM_ENVS"
)

if [[ "${RESUME:-0}" == "1" ]] || [[ -n "${LOAD_RUN:-}" ]]; then
  OMNI_RUN="$(ls -1d "$EXP_DIR"/2026-* 2>/dev/null | sort | tail -1 || true)"
  LOAD_RUN="${LOAD_RUN:-$(basename "$OMNI_RUN")}"
  if [[ -z "${CHECKPOINT:-}" ]] && [[ -n "$OMNI_RUN" ]]; then
    CHECKPOINT="$(ls -1 "$OMNI_RUN"/model_*.pt 2>/dev/null | sort -V | tail -1 | xargs -n1 basename)"
  fi
  CMD+=(--resume --load_run="$LOAD_RUN")
  [[ -n "${CHECKPOINT:-}" ]] && CMD+=(--checkpoint="$CHECKPOINT")
  echo "[stairs-omni] RESUME $LOAD_RUN/${CHECKPOINT:-latest}"
else
  echo "[stairs-omni] FRESH train with height-scan (239D) — not resuming flat 52D"
fi

[[ -n "${MAX_ITERS:-}" ]] && CMD+=(--max_iterations="$MAX_ITERS")
if [[ "$GUI" != "1" ]]; then
  CMD+=(--headless)
  echo "[stairs-omni] headless task=$TASK num_envs=$NUM_ENVS"
else
  echo "[stairs-omni] GUI task=$TASK num_envs=$NUM_ENVS"
fi
CMD+=("$@")
echo "[stairs-omni] ${CMD[*]}"
exec "${CMD[@]}"
