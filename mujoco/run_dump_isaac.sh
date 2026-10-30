#!/usr/bin/env bash
set -euo pipefail
ROOT_PARK=/home/hi/liu/extreme-quadruped-parkour
ILAB=/home/hi/liu/wheeled_legged_dog/IsaacLab
source /home/hi/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab51
export ZSH_VERSION="" OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
set +u
source "$ILAB/_isaac_sim/setup_conda_env.sh"
set -euo pipefail
export PYTHONPATH="\
${ROOT_PARK}/source/isaaclab_tasks:\
${ROOT_PARK}/source/isaaclab_assets:\
${ROOT_PARK}/source/isaaclab_rl:\
${ROOT_PARK}/source/isaaclab:\
${ROOT_PARK}/scripts/reinforcement_learning/rsl_rl:\
${PYTHONPATH:-}"
cd /home/hi/liu/diandog/mujoco
exec python -u dump_isaac_obs.py
