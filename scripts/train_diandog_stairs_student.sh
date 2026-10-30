#!/usr/bin/env bash
# DianDog 楼梯盲学生蒸馏：52D 本体观测，DAgger 跟学 239D scanned teacher。
#   GUI=0 NUM_ENVS=4096 bash scripts/train_diandog_stairs_student.sh
#   TEACHER_CHECKPOINT=/path/to/model_XXXX.pt bash scripts/train_diandog_stairs_student.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WDOG="/home/hi/liu/wheeled_legged_dog"
ILAB="$WDOG/IsaacLab"

TEACHER_EXP="$ROOT/logs/rsl_rl/diandog_stairs_omni"
STUDENT_EXP="$ROOT/logs/rsl_rl/diandog_stairs_omni_student"
mkdir -p "$STUDENT_EXP"

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
TASK="${TASK:-DianDog-Stairs-Omni-Student-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
GUI="${GUI:-0}"
MAX_ITERS="${MAX_ITERS:-8000}"

# Resolve teacher checkpoint (default: latest finished omni teacher)
if [[ -z "${TEACHER_CHECKPOINT:-}" ]]; then
  TEACHER_RUN="${TEACHER_RUN:-$(ls -1d "$TEACHER_EXP"/2026-* 2>/dev/null | sort | tail -1 || true)}"
  if [[ -z "$TEACHER_RUN" ]]; then
    echo "[stairs-student] ERROR: no teacher run under $TEACHER_EXP"
    exit 1
  fi
  TEACHER_CHECKPOINT="$(ls -1 "$TEACHER_RUN"/model_*.pt 2>/dev/null | sort -V | tail -1)"
fi
if [[ -z "${TEACHER_CHECKPOINT:-}" ]] || [[ ! -f "$TEACHER_CHECKPOINT" ]]; then
  echo "[stairs-student] ERROR: teacher checkpoint missing: ${TEACHER_CHECKPOINT:-}"
  exit 1
fi
export TEACHER_CHECKPOINT
export TEACHER_LOG_ROOT="$TEACHER_EXP"

# Optional: pin load_run/checkpoint for get_checkpoint_path fallback
LOAD_RUN="${LOAD_RUN:-$(basename "$(dirname "$TEACHER_CHECKPOINT")")}"
CHECKPOINT="${CHECKPOINT:-$(basename "$TEACHER_CHECKPOINT")}"

CMD=(
  python -u scripts/reinforcement_learning/rsl_rl/train.py
  --task="$TASK"
  --num_envs="$NUM_ENVS"
  --max_iterations="$MAX_ITERS"
  --load_run="$LOAD_RUN"
  --checkpoint="$CHECKPOINT"
)
if [[ "$GUI" != "1" ]]; then
  CMD+=(--headless)
fi
CMD+=("$@")

echo "[stairs-student] teacher=$TEACHER_CHECKPOINT"
echo "[stairs-student] task=$TASK num_envs=$NUM_ENVS max_iters=$MAX_ITERS gui=$GUI"
echo "[stairs-student] ${CMD[*]}"
exec "${CMD[@]}"
