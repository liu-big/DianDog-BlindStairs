#!/usr/bin/env bash
# DianDog MuJoCo teleop after flat train.
#   bash /home/hi/liu/diandog/mujoco/play.sh
#   bash /home/hi/liu/diandog/mujoco/play.sh /path/to/model_4999.pt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CKPT="${1:-/home/hi/liu/extreme-quadruped-parkour/logs/rsl_rl/diandog_flat_direct/2026-10-27_22-33-10/model_4999.pt}"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/home/hi/.Xauthority}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Prefer glfw for interactive window when display is available
if [[ -n "${DISPLAY:-}" ]]; then
  export MUJOCO_GL=glfw
fi
source /home/hi/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab51
# torch lives in Isaac Sim pip_prebundle (not conda site-packages)
# shellcheck source=/dev/null
source /home/hi/isaacsim/setup_python_env.sh
cd "$ROOT"
echo "[diandog-mujoco] ckpt=$CKPT"
echo "[diandog-mujoco] 小键盘: 8/2=vx 4/6=yaw 7/9=vy 0/5=停 | R=重置 P=暂停"
# drop already-consumed ckpt path from "$@"
shift || true
exec python -u play_diandog_mujoco.py --ckpt "$CKPT" "$@"
