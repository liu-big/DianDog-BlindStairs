#!/usr/bin/env bash
# Isaac-aligned DianDog MuJoCo teleop — large bright ground + 15cm/20cm stairs.
#   CKPT=/path/to/model.pt bash play_stairs.sh
#   bash play_stairs.sh --auto-fwd --vx 0.7
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CKPT="${CKPT:-}"
TREAD="${TREAD:-0.30}"
STEPS="${STEPS:-8}"
START_LEN="${START_LEN:-8.0}"
END_LEN="${END_LEN:-6.0}"
CHAMFER="${CHAMFER:-0.05}"
FOOT_R="${FOOT_R:-0.020}"
NO_HFE="${NO_HFE:-1}"
NO_KFE="${NO_KFE:-0}"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"

EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --step-h) shift 2 ;;  # ignored: dual 15+20 always
    --tread) TREAD="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --start-len) START_LEN="$2"; shift 2 ;;
    --end-len) END_LEN="$2"; shift 2 ;;
    --chamfer) CHAMFER="$2"; shift 2 ;;
    --foot-r) FOOT_R="$2"; shift 2 ;;
    --no-hfe) NO_HFE=1; shift ;;
    --with-hfe) NO_HFE=0; shift ;;
    --no-kfe) NO_KFE=1; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ -z "${CKPT}" || ! -f "$CKPT" ]]; then
  CKPT="$(ls -1t /home/hi/liu/extreme-quadruped-parkour/logs/rsl_rl/diandog_stairs_omni_student_rl/*/model_*.pt 2>/dev/null | head -1 || true)"
fi
if [[ -z "${CKPT}" || ! -f "$CKPT" ]]; then
  CKPT="$(ls -1t /home/hi/liu/extreme-quadruped-parkour/logs/rsl_rl/diandog_stairs_omni_student/*/model_*.pt 2>/dev/null | head -1 || true)"
fi
if [[ -z "${CKPT}" || ! -f "$CKPT" ]]; then
  echo "[stairs-mujoco] no checkpoint found" >&2
  exit 1
fi

source /home/hi/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab51
export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
set +u
if [[ -f /home/hi/isaacsim/setup_python_env.sh ]]; then
  # shellcheck source=/dev/null
  source /home/hi/isaacsim/setup_python_env.sh
elif [[ -f /home/hi/liu/wheeled_legged_dog/IsaacLab/_isaac_sim/setup_conda_env.sh ]]; then
  # shellcheck source=/dev/null
  source /home/hi/liu/wheeled_legged_dog/IsaacLab/_isaac_sim/setup_conda_env.sh
fi
set -euo pipefail

cd "$ROOT"
GEN=(
  python -u gen_stairs_xml.py --dual
  --tread "$TREAD" --steps "$STEPS"
  --start-len "$START_LEN" --end-len "$END_LEN"
  --chamfer "$CHAMFER" --foot-r "$FOOT_R"
  -o "$ROOT/diandog_stairs.xml"
)
[[ "$NO_HFE" == "1" ]] && GEN+=(--no-hfe)
[[ "$NO_KFE" == "1" ]] && GEN+=(--no-kfe)
"${GEN[@]}"
echo "[stairs-mujoco] dual strips: 15cm (+Y, blue marker) | 20cm (-Y, orange marker)"
echo "[stairs-mujoco] ckpt=$CKPT"
echo "[stairs-mujoco] 8/2=vx 7/9=vy 4/6=yaw 0/5=stop | R=reset P=pause"
exec python -u play_diandog_stairs_mujoco.py \
  --scene "$ROOT/diandog_stairs.xml" \
  --ckpt "$CKPT" \
  "${EXTRA[@]}"
