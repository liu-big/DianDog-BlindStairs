#!/usr/bin/env python3
"""Headless dump of DianDog Isaac joint order + policy obs for MuJoCo alignment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app = AppLauncher(args_cli).app

import gymnasium as gym
import torch

ROOT = Path("/home/hi/liu/extreme-quadruped-parkour")
sys.path[:0] = [
    str(ROOT / "source/isaaclab_tasks"),
    str(ROOT / "source/isaaclab_assets"),
    str(ROOT / "source/isaaclab_rl"),
    str(ROOT / "source/isaaclab"),
    str(ROOT / "scripts/reinforcement_learning/rsl_rl"),
]

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from train import _convert_rsl_rl_cfg_dict

CKPT = ROOT / "logs/rsl_rl/diandog_flat_direct/2026-10-27_22-33-10/model_4999.pt"


def main() -> None:
    env_cfg = parse_env_cfg("DianDog-Direct-v0", device="cuda:0", num_envs=1)
    agent_cfg = load_cfg_from_registry("DianDog-Direct-v0", "rsl_rl_cfg_entry_point")
    env = gym.make("DianDog-Direct-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    robot = env.unwrapped._robot

    print("JOINT_NAMES", list(robot.data.joint_names))
    print("BODY_NAMES", list(robot.data.body_names))
    print("default_joint_pos", robot.data.default_joint_pos[0].detach().cpu().numpy().tolist())
    print("feet_ids", env.unwrapped._feet_ids)

    cfg = _convert_rsl_rl_cfg_dict(agent_cfg.to_dict())
    runner = OnPolicyRunner(env, cfg, log_dir=None, device="cuda:0")
    runner.load(str(CKPT))
    policy = runner.get_inference_policy(device="cuda:0")

    obs = env.get_observations()
    env.unwrapped._commands[:] = torch.tensor([[0.5, 0.0, 0.0]], device="cuda:0")
    for _ in range(30):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    pol = obs["policy"][0].detach().cpu().numpy()
    print("OBS_DIM", int(pol.shape[0]))
    print("OBS", pol.tolist())
    print("CMD", env.unwrapped._commands[0].detach().cpu().numpy().tolist())
    print("joint_pos", robot.data.joint_pos[0].detach().cpu().numpy().tolist())
    print("joint_vel", robot.data.joint_vel[0].detach().cpu().numpy().tolist())
    print("proj_grav", robot.data.projected_gravity_b[0].detach().cpu().numpy().tolist())
    print("lin_vel_b", robot.data.root_lin_vel_b[0].detach().cpu().numpy().tolist())
    print("ang_vel_b", robot.data.root_ang_vel_b[0].detach().cpu().numpy().tolist())
    print("root_z", float(robot.data.root_pos_w[0, 2].item()))
    print("last_action", env.unwrapped._actions[0].detach().cpu().numpy().tolist())
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
