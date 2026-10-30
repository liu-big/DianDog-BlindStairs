#!/usr/bin/env bash
# Blind 52D PPO finetune after distill — active foot lift / anti shin-jam.
# Scan stays reward+critic only (not in policy).
#   GUI=1 NUM_ENVS=4096 bash scripts/train_diandog_stairs_student_rl.sh
#   STUDENT_CHECKPOINT=/path/to/model_8399.pt bash scripts/train_diandog_stairs_student_rl.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WDOG="/home/hi/liu/wheeled_legged_dog"
ILAB="$WDOG/IsaacLab"

STUDENT_EXP="$ROOT/logs/rsl_rl/diandog_stairs_omni_student"
RL_EXP="$ROOT/logs/rsl_rl/diandog_stairs_omni_student_rl"
mkdir -p "$RL_EXP"

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
${PYTHONPATH:-}"

cd "$ROOT"
TASK="${TASK:-DianDog-Stairs-Omni-Student-RL-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
GUI="${GUI:-1}"
MAX_ITERS="${MAX_ITERS:-10000}"

CMD=(
  python -u scripts/reinforcement_learning/rsl_rl/train.py
  --task="$TASK"
  --num_envs="$NUM_ENVS"
  --max_iterations="$MAX_ITERS"
)

# RESUME=1 → continue PPO run under student_rl; else init actor from distill student
if [[ "${RESUME:-0}" == "1" ]]; then
  unset STUDENT_CHECKPOINT || true
  RL_RUN="${LOAD_RUN:-$(ls -1d "$RL_EXP"/2026-* 2>/dev/null | sort | tail -1 | xargs -n1 basename || true)}"
  if [[ -z "$RL_RUN" ]]; then
    echo "[student-rl] ERROR: no RL run to resume under $RL_EXP" >&2
    exit 1
  fi
  if [[ -z "${CHECKPOINT:-}" ]]; then
    CHECKPOINT="$(ls -1 "$RL_EXP/$RL_RUN"/model_*.pt 2>/dev/null | sort -V | tail -1 | xargs -n1 basename)"
  fi
  CMD+=(--resume --load_run="$RL_RUN")
  [[ -n "${CHECKPOINT:-}" ]] && CMD+=(--checkpoint="$CHECKPOINT")
  echo "[student-rl] RESUME $RL_RUN/${CHECKPOINT:-latest}"
else
  if [[ -z "${STUDENT_CHECKPOINT:-}" || ! -f "${STUDENT_CHECKPOINT}" ]]; then
    STUDENT_CHECKPOINT="$(ls -1t "$STUDENT_EXP"/*/model_*.pt 2>/dev/null | head -1 || true)"
  fi
  if [[ -z "${STUDENT_CHECKPOINT:-}" || ! -f "$STUDENT_CHECKPOINT" ]]; then
    echo "[student-rl] ERROR: no distilled student checkpoint under $STUDENT_EXP" >&2
    exit 1
  fi
  export STUDENT_CHECKPOINT
  echo "[student-rl] init_from=$STUDENT_CHECKPOINT"
fi

if [[ "$GUI" != "1" ]]; then
  CMD+=(--headless)
fi
CMD+=("$@")

echo "[student-rl] task=$TASK num_envs=$NUM_ENVS max_iters=$MAX_ITERS gui=$GUI"
echo "[student-rl] ${CMD[*]}"
exec "${CMD[@]}"
