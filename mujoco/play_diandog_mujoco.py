#!/usr/bin/env python3
"""DianDog flat policy → MuJoCo teleop (同轮组小键盘).

Obs (52D, matches Go2 Direct / DianDog-Direct-v0):
  joint_pos_rel(12), joint_vel(12), proj_grav(3),
  root_lin_vel_b(3), root_ang_vel_b(3), cmd(3), last_action(12), foot_contact(4)

键位（焦点在 MuJoCo 窗口，与 wheeldog 一致）:
  8/2  前进/后退 (vx)
  4/6  左转/右转 (yaw)
  7/9  左移/右移 (vy)
  0/5  停止（指令清零 → 站立）
  R    重置
  P    暂停
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE = ROOT / "diandog_flat.xml"
DEFAULT_CKPT = Path(
    "/home/hi/liu/extreme-quadruped-parkour/logs/rsl_rl/diandog_flat_direct/"
    "2026-10-27_22-33-10/model_4999.pt"
)

# Isaac PhysX joint_names (haa×4, hfe×4, kfe×4) — kit log
JOINTS = [
    "fl_haa_j1", "fr_haa_j1", "rl_haa_j1", "rr_haa_j1",
    "fl_hfe_j2", "fr_hfe_j2", "rl_hfe_j2", "rr_hfe_j2",
    "fl_kfe_j3", "fr_kfe_j3", "rl_kfe_j3", "rr_kfe_j3",
]
FEET = ["fl_ft", "fr_ft", "rl_ft", "rr_ft"]
DEFAULT_Q = np.array(
    [0.03, -0.03, -0.03, 0.03, -0.18, 0.18, -0.18, 0.18, -0.30, 0.30, -0.30, 0.30],
    dtype=np.float64,
)
ACTION_SCALE = 0.28
OBS_DIM, ACT_DIM = 52, 12
DECIMATION = 4
SIM_DT = 0.005
KP, KD = 80.0, 3.5
NUMPAD_STEP = 0.1
CMD_LIMITS = [(-1.5, 1.5), (-0.6, 0.6), (-1.0, 1.0)]
CMD_DEADZONE = 0.05  # |cmd|<此 → 纯站立，不跑策略

# GLFW: keypad 0-9 = 320..329；主键盘数字亦可
_KP = {i: 320 + i for i in range(10)}
_DIGIT = {i: ord("0") + i for i in range(10)}


def _codes(*ds: int) -> set[int]:
    out: set[int] = set()
    for d in ds:
        out.add(_KP[d])
        out.add(_DIGIT[d])
    return out


KEY_STOP = _codes(0, 5)
KEY_MAP = {
    **{c: (0, +1.0) for c in _codes(8)},  # vx+
    **{c: (0, -1.0) for c in _codes(2)},  # vx-
    **{c: (2, +1.0) for c in _codes(4)},  # yaw+
    **{c: (2, -1.0) for c in _codes(6)},  # yaw-
    **{c: (1, +1.0) for c in _codes(7)},  # vy+
    **{c: (1, -1.0) for c in _codes(9)},  # vy-
}


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz / max(np.linalg.norm(q_wxyz), 1e-12)
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return R.T @ v


class ActorMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(52, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 12),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


def load_actor(ckpt: Path) -> tuple[ActorMLP, np.ndarray, np.ndarray]:
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    sd = blob["actor_state_dict"]
    actor = ActorMLP()
    actor.load_state_dict({k: v for k, v in sd.items() if k.startswith("mlp.")}, strict=True)
    actor.eval()
    mean = sd["obs_normalizer._mean"].squeeze(0).numpy().astype(np.float32)
    std = sd["obs_normalizer._std"].squeeze(0).numpy().astype(np.float32)
    # rsl-rl EmpiricalNormalization: (x - mean) / (std + eps), eps default 1e-2
    print(f"[policy] {ckpt}  iter={blob.get('iter', '?')}", flush=True)
    return actor, mean, std


class DianDogController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, ckpt: Path):
        self.model = model
        self.data = data
        self.actor, self.obs_mean, self.obs_std = load_actor(ckpt)
        self.command = np.zeros(3, dtype=np.float64)
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)

        joint_to_act = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
        self.qpos_adr, self.qvel_adr, self.act_ids = [], [], []
        for name in JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"missing joint {name}")
            self.qpos_adr.append(int(model.jnt_qposadr[jid]))
            self.qvel_adr.append(int(model.jnt_dofadr[jid]))
            self.act_ids.append(joint_to_act[jid])
        self.qpos_adr = np.asarray(self.qpos_adr)
        self.qvel_adr = np.asarray(self.qvel_adr)
        self.act_ids = np.asarray(self.act_ids)
        self.foot_body = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in FEET
        ]
        if any(b < 0 for b in self.foot_body):
            raise RuntimeError(f"foot bodies missing: {self.foot_body}")

    def reset(self) -> None:
        self.last_action[:] = 0.0
        self.command[:] = 0.0
        mujoco.mj_resetData(self.model, self.data)
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, int(kid))
        else:
            self.data.qpos[0:3] = [0.0, 0.0, 0.48]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_adr] = DEFAULT_Q
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = DEFAULT_Q
        mujoco.mj_forward(self.model, self.data)
        for _ in range(200):
            self.apply_actuators()
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.last_action[:] = 0.0
        self.command[:] = 0.0
        self.apply_actuators()

    def foot_contacts(self) -> np.ndarray:
        """1 if foot body appears in any active contact (matches train force>1 roughly)."""
        out = np.zeros(4, dtype=np.float32)
        foot_set = {int(b): i for i, b in enumerate(self.foot_body)}
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            for gid in (c.geom1, c.geom2):
                bid = int(self.model.geom_bodyid[gid])
                if bid in foot_set:
                    out[foot_set[bid]] = 1.0
        return out

    def build_obs(self) -> np.ndarray:
        q = self.data.qpos[self.qpos_adr].astype(np.float64)
        dq = self.data.qvel[self.qvel_adr].astype(np.float64)
        quat = self.data.qpos[3:7].astype(np.float64)
        # freejoint qvel: [ang(3), lin(3)] in MuJoCo for free joint? 
        # Actually freejoint: qvel[0:3]=lin world? No — MuJoCo freejoint dof order is
        # 3 translational then 3 rotational in the BODY frame for newer mujoco?
        # Standard: qvel for freejoint is [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z] in world? 
        # MuJoCo docs: free joint qvel is 3 translational (world) + 3 rotational (body-local).
        lin_w = self.data.qvel[0:3].astype(np.float64)
        ang_b = self.data.qvel[3:6].astype(np.float64)
        lin_b = quat_rotate_inverse(quat, lin_w)
        grav = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        obs = np.concatenate(
            [
                (q - DEFAULT_Q).astype(np.float32),
                dq.astype(np.float32),
                grav.astype(np.float32),
                lin_b.astype(np.float32),
                ang_b.astype(np.float32),
                self.command.astype(np.float32),
                self.last_action.copy(),
                self.foot_contacts(),
            ]
        )
        if obs.shape != (OBS_DIM,):
            raise RuntimeError(f"obs {obs.shape} != {OBS_DIM}")
        # Match rsl-rl EmpiricalNormalization exactly (eps=1e-2).
        obs = (obs - self.obs_mean) / (self.obs_std + 1e-2)
        return obs.astype(np.float32)

    def apply_actuators(self) -> None:
        # Isaac DelayedPD: tau = Kp*(q_des-q) - Kd*dq, effort ±40
        tgt = DEFAULT_Q + ACTION_SCALE * self.last_action.astype(np.float64)
        q = self.data.qpos[self.qpos_adr]
        dq = self.data.qvel[self.qvel_adr]
        tau = KP * (tgt - q) - KD * dq
        for i, aid in enumerate(self.act_ids):
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = float(np.clip(tau[i], lo, hi))

    def policy_step(self) -> None:
        if float(np.linalg.norm(self.command)) < CMD_DEADZONE:
            self.last_action[:] = 0.0
            self.apply_actuators()
            return
        obs = self.build_obs()
        with torch.no_grad():
            action = self.actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        # Flat policy actions stay roughly in [-2, 2] in training; hard clip avoids tip-over.
        self.last_action = np.clip(action, -2.0, 2.0).astype(np.float32)
        self.apply_actuators()

    def status(self) -> str:
        return (
            f"cmd=({self.command[0]:+.2f},{self.command[1]:+.2f},{self.command[2]:+.2f}) "
            f"z={self.data.qpos[2]:.3f} |feet={self.foot_contacts().sum():.0f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--duration", type=float, default=0.0)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene.expanduser().resolve()))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    ctrl = DianDogController(model, data, args.ckpt.expanduser().resolve())
    ctrl.reset()
    print(f"[diandog] settled z={data.qpos[2]:.3f}  {ctrl.status()}", flush=True)
    print(
        "[diandog] 8/2=vx  4/6=yaw  7/9=vy  0/5=停 | R=重置 P=暂停（同轮组）",
        flush=True,
    )
    print("[diandog] 默认站立；按 8 等发指令后策略才动腿", flush=True)

    paused = {"v": False}

    def on_key(keycode: int) -> None:
        if keycode in (ord("P"), ord("p")):
            paused["v"] = not paused["v"]
            print("paused" if paused["v"] else "running", flush=True)
            return
        if keycode in (ord("R"), ord("r")):
            ctrl.reset()
            print(f"reset | {ctrl.status()}", flush=True)
            return
        if keycode in KEY_STOP:
            ctrl.command[:] = 0.0
            ctrl.last_action[:] = 0.0
        elif keycode in KEY_MAP:
            axis, sign = KEY_MAP[keycode]
            lo, hi = CMD_LIMITS[axis]
            ctrl.command[axis] = float(
                np.clip(ctrl.command[axis] + sign * NUMPAD_STEP, lo, hi)
            )
        else:
            return
        print(ctrl.status(), flush=True)

    phys_i = 0
    t0 = time.time()
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.distance = 2.5
        while viewer.is_running():
            if args.duration > 0 and time.time() - t0 > args.duration:
                break
            step_t = time.time()
            if not paused["v"]:
                if phys_i % DECIMATION == 0:
                    ctrl.policy_step()
                else:
                    ctrl.apply_actuators()
                mujoco.mj_step(model, data)
                phys_i += 1
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()
            dt = model.opt.timestep - (time.time() - step_t)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
