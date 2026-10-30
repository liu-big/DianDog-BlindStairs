#!/usr/bin/env bash
# Play DianDog stairs-omni in Isaac with keyboard teleop.
#   bash scripts/play_diandog_stairs_omni.sh
#   CKPT=/path/to/model.pt NUM_ENVS=4 bash scripts/play_diandog_stairs_omni.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WDOG="/home/hi/liu/wheeled_legged_dog"
ILAB="$WDOG/IsaacLab"
CKPT="${CKPT:-}"
NUM_ENVS="${NUM_ENVS:-4}"
TASK="${TASK:-}"

if [[ -z "${CKPT}" || ! -f "$CKPT" ]]; then
  CKPT="$(ls -1t "$ROOT"/logs/rsl_rl/diandog_stairs_omni_student_rl/*/model_*.pt 2>/dev/null | head -1 || true)"
fi
if [[ -z "${CKPT}" || ! -f "$CKPT" ]]; then
  CKPT="$(ls -1t "$ROOT"/logs/rsl_rl/diandog_stairs_omni_student/*/model_*.pt 2>/dev/null | head -1 || true)"
fi
if [[ -z "${CKPT}" || ! -f "$CKPT" ]]; then
  CKPT="$(ls -1t "$ROOT"/logs/rsl_rl/diandog_stairs_omni/*/model_*.pt 2>/dev/null | head -1 || true)"
fi
if [[ -z "${CKPT}" || ! -f "$CKPT" ]]; then
  echo "[stairs-play] no checkpoint found" >&2
  exit 1
fi
# Auto-pick blind play task from ckpt path
if [[ -z "${TASK}" ]]; then
  if [[ "$CKPT" == *diandog_stairs_omni_student_rl* ]]; then
    TASK="DianDog-Stairs-Omni-Student-RL-Play-v0"
  elif [[ "$CKPT" == *diandog_stairs_omni_student* ]]; then
    TASK="DianDog-Stairs-Omni-Student-Play-v0"
  else
    TASK="DianDog-Stairs-Omni-Play-v0"
  fi
fi

source /home/hi/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab51
export ZSH_VERSION=""
export OMNI_KIT_ACCEPT_EULA=YES
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export PYTHONUNBUFFERED=1
set +u
source "$ILAB/_isaac_sim/setup_conda_env.sh"
set -euo pipefail

export PYTHONPATH="\
${ROOT}/source/isaaclab_tasks:\
${ROOT}/source/isaaclab_assets:\
${ROOT}/source/isaaclab_rl:\
${ROOT}/source/isaaclab:\
${ROOT}/scripts/reinforcement_learning/rsl_rl:\
${PYTHONPATH:-}"

cd "$ROOT"
echo "[stairs-play] Isaac GUI teleop task=$TASK ckpt=$CKPT num_envs=$NUM_ENVS"
echo "[stairs-play] 焦点在视口: ↑↓←→ / 小键盘 控制；L=清零指令"
exec python -u scripts/play_diandog_stairs_omni_teleop.py \
  --task "$TASK" \
  --num_envs "$NUM_ENVS" \
  --checkpoint "$CKPT" \
  --real-time
