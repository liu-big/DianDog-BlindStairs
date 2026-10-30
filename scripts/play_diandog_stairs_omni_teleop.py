#!/usr/bin/env python3
"""Isaac Stairs-Omni play with keyboard SE(2) velocity commands.

Focus the Isaac Sim viewport, then:
  ↑ / Numpad8  forward     ↓ / Numpad2  back
  → / Numpad4  right       ← / Numpad6  left
  Z / Numpad7  yaw+        X / Numpad9  yaw-
  L            zero command
"""
from __future__ import annotations

import argparse
import builtins
import os
import sys
import time

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reinforcement_learning", "rsl_rl"))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="DianDog Stairs-Omni keyboard play")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default="DianDog-Stairs-Omni-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--real-time", action="store_true", default=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _register_custom_actor() -> None:
    try:
        from isaaclab_tasks.direct.go2.agents.actor_critic_scan import ActorCriticScan

        builtins.ActorCriticScan = ActorCriticScan
    except Exception:
        pass


def _convert_rsl_rl_cfg_dict(cfg_dict: dict) -> dict:
    if "policy" not in cfg_dict:
        return cfg_dict
    policy = cfg_dict.pop("policy", {})
    init_noise_std = policy.pop("init_noise_std", 1.0)
    noise_std_type = policy.pop("noise_std_type", "scalar")
    distribution_cfg = {
        "class_name": "GaussianDistribution",
        "init_std": init_noise_std,
        "std_type": noise_std_type,
    }
    actor_obs_norm = policy.pop("actor_obs_normalization", False)
    critic_obs_norm = policy.pop("critic_obs_normalization", False)
    # Distill student cfg uses student_* fields
    if "student_hidden_dims" in policy:
        actor_obs_norm = policy.pop("student_obs_normalization", actor_obs_norm)
        critic_obs_norm = policy.pop("teacher_obs_normalization", critic_obs_norm)
    empirical_norm = cfg_dict.pop("empirical_normalization", None)
    if empirical_norm is not None:
        actor_obs_norm = empirical_norm
        critic_obs_norm = empirical_norm
    for k in (
        "class_name",
        "state_dependent_std",
        "num_prop_obs",
        "num_scan_obs",
        "num_actor_scan_obs",
        "num_critic_scan_obs",
        "scan_encoder_dims",
        "actor_scan_encoder_dims",
        "critic_scan_encoder_dims",
        "encode_scan_for_critic",
        "priv_obs_encoder_dims",
        "priv_encoder_dims",
        "student_obs_normalization",
        "teacher_obs_normalization",
    ):
        policy.pop(k, None)
    actor_hidden_dims = policy.pop(
        "actor_hidden_dims", policy.pop("student_hidden_dims", [256, 256, 256])
    )
    critic_hidden_dims = policy.pop(
        "critic_hidden_dims", policy.pop("teacher_hidden_dims", actor_hidden_dims)
    )
    activation = policy.pop("activation", "elu")
    # Play only needs actor; force OnPolicyRunner/PPO layout
    cfg_dict["class_name"] = "OnPolicyRunner"
    cfg_dict["algorithm"] = {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.0,
        "num_learning_epochs": 1,
        "num_mini_batches": 1,
        "learning_rate": 1e-4,
        "schedule": "fixed",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    }
    cfg_dict["actor"] = {
        "class_name": "MLPModel",
        "hidden_dims": actor_hidden_dims,
        "activation": activation,
        "obs_normalization": actor_obs_norm,
        "distribution_cfg": distribution_cfg,
    }
    cfg_dict["critic"] = {
        "class_name": "MLPModel",
        "hidden_dims": critic_hidden_dims,
        "activation": activation,
        "obs_normalization": critic_obs_norm,
    }
    cfg_dict["obs_groups"] = {"actor": ["policy"], "critic": ["critic"]}
    return cfg_dict


def _load_play_checkpoint(runner: OnPolicyRunner, path: str) -> None:
    """Load PPO actor or distilled student into the play runner (actor only)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "student_state_dict" in blob:
        sd = blob["student_state_dict"]
        src = "student"
    elif "actor_state_dict" in blob:
        sd = blob["actor_state_dict"]
        src = "actor"
    else:
        raise KeyError(f"no actor/student weights in {path}; keys={list(blob.keys())}")
    # Play only needs the policy; skip critic (teacher 239D ≠ critic 268D).
    missing, unexpected = runner.alg.actor.load_state_dict(sd, strict=False)
    runner.alg.actor.eval()
    w0 = sd["mlp.0.weight"]
    print(
        f"[stairs-teleop] loaded {src}→actor from {path}  "
        f"obs_in={tuple(w0.shape)} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    _register_custom_actor()

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # teleop: full omni; don't force stairs-only forward
    if hasattr(env_cfg, "command_stairs_forward_only"):
        env_cfg.command_stairs_forward_only = False
    env_cfg.command_mode = "fixed"
    env_cfg.fixed_command = (0.0, 0.0, 0.0)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[stairs-teleop] ckpt={resume_path} num_envs={args_cli.num_envs}", flush=True)
    train_cfg = _convert_rsl_rl_cfg_dict(agent_cfg.to_dict())
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    keyboard = Se2Keyboard(
        Se2KeyboardCfg(
            v_x_sensitivity=0.7,
            v_y_sensitivity=0.4,
            omega_z_sensitivity=0.8,
            sim_device=str(env.unwrapped.device),
        )
    )
    print(keyboard, flush=True)
    print("[stairs-teleop] Focus Isaac viewport → arrows/numpad drive; L=stop", flush=True)

    raw = env.unwrapped
    dt = raw.step_dt
    obs = env.get_observations()

    while simulation_app.is_running():
        t0 = time.time()
        cmd = keyboard.advance().to(device=raw.device, dtype=raw._commands.dtype).view(1, 3)
        raw._commands[:] = cmd.expand(raw.num_envs, -1)

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        sleep = dt - (time.time() - t0)
        if args_cli.real_time and sleep > 0:
            time.sleep(sleep)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
