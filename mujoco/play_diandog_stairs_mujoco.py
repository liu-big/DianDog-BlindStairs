#!/usr/bin/env python3
"""DianDog Isaac checkpoint → MuJoCo teleop (52D / 239D).

Aligned to Go2/DianDog DirectRL env (go2_env.py):
  obs = [q-q0(12), dq(12), grav_b(3), lin_vel_b(3), ang_vel_b(3),
         cmd(3), last_action(12), foot_contact(4)]  (+ optional 187 scan)
  q_des = q0 + action_scale * action
  τ = KP*(q_des-q) - KD*dq   (matches DelayedPDActuator stiffness/damping)
  sim dt=0.005, decimation=4  (Isaac sim.dt=1/200)

Keys (focus MuJoCo window):
  8/2  vx+/-     7/9  vy+/-     4/6  yaw+/-
  0/5  stop      R reset       P pause
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent

# Isaac PhysX joint_names order (NOT MuJoCo XML tree order)
JOINTS = [
    "fl_haa_j1", "fr_haa_j1", "rl_haa_j1", "rr_haa_j1",
    "fl_hfe_j2", "fr_hfe_j2", "rl_hfe_j2", "rr_hfe_j2",
    "fl_kfe_j3", "fr_kfe_j3", "rl_kfe_j3", "rr_kfe_j3",
]
# ContactSensor find_bodies(".*_ft") follows USD body order ≈ fl,fr,rl,rr
FEET = ["fl_ft", "fr_ft", "rl_ft", "rr_ft"]

# diandog.py _STAND — must match training default_joint_pos
DEFAULT_Q = np.array(
    [0.04, -0.04, -0.04, 0.04, -0.25, 0.25, -0.25, 0.25, -0.40, 0.40, -0.40, 0.40],
    dtype=np.float64,
)

ACTION_SCALE = 0.3
PROP_DIM, SCAN_DIM, ACT_DIM = 52, 187, 12
DECIMATION = 4
SIM_DT = 0.005  # 1/200
KP, KD = 80.0, 3.5  # DelayedPDActuatorCfg
OBS_NORM_EPS = 1e-2  # rsl-rl EmpiricalNormalization

CMD_STEP = 0.10
CMD_LIMITS = np.array([[-1.2, 1.2], [-0.7, 0.7], [-1.2, 1.2]], dtype=np.float64)

SCAN_SIZE = (1.6, 1.0)
SCAN_RES = 0.1
SCAN_Z_OFFSET = 20.0

_KP = {i: 320 + i for i in range(10)}
_DIGIT = {i: ord("0") + i for i in range(10)}


def _codes(*ds: int) -> set[int]:
    out: set[int] = set()
    for d in ds:
        out.add(_KP[d])
        out.add(_DIGIT[d])
    return out


KEY_STOP = _codes(0, 5)
# axis index, sign
KEY_DELTA: dict[int, tuple[int, float]] = {
    **{c: (0, +1.0) for c in _codes(8)},
    **{c: (0, -1.0) for c in _codes(2)},
    **{c: (1, +1.0) for c in _codes(7)},
    **{c: (1, -1.0) for c in _codes(9)},
    **{c: (2, +1.0) for c in _codes(4)},
    **{c: (2, -1.0) for c in _codes(6)},
}


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz / max(np.linalg.norm(q_wxyz), 1e-12)
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R.T @ v


def yaw_from_quat(q_wxyz: np.ndarray) -> float:
    w, x, y, z = q_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def make_scan_grid() -> np.ndarray:
    x = np.arange(-SCAN_SIZE[0] / 2, SCAN_SIZE[0] / 2 + 1e-9, SCAN_RES)
    y = np.arange(-SCAN_SIZE[1] / 2, SCAN_SIZE[1] / 2 + 1e-9, SCAN_RES)
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    pts = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    assert pts.shape[0] == SCAN_DIM
    return pts.astype(np.float64)


class ActorMLP(nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, ACT_DIM),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


def load_actor(ckpt: Path) -> tuple[ActorMLP, np.ndarray, np.ndarray, int, str]:
    torch.set_num_threads(1)
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if "actor_state_dict" in blob:
        sd, src = blob["actor_state_dict"], "actor"
    elif "student_state_dict" in blob:
        sd, src = blob["student_state_dict"], "student"
    else:
        raise KeyError(f"no actor/student in {ckpt}; keys={list(blob.keys())}")
    w0 = sd["mlp.0.weight"]
    obs_dim = int(w0.shape[1])
    actor = ActorMLP(obs_dim)
    actor.load_state_dict({k: v for k, v in sd.items() if k.startswith("mlp.")}, strict=True)
    actor.eval()
    mean = sd["obs_normalizer._mean"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    std = sd["obs_normalizer._std"].squeeze(0).detach().cpu().numpy().astype(np.float32)
    assert mean.shape[0] == obs_dim and std.shape[0] == obs_dim
    print(
        f"[policy] {ckpt}  src={src}  iter={blob.get('iter', '?')}  "
        f"obs_dim={obs_dim}  action_scale={ACTION_SCALE}  kp={KP} kd={KD}",
        flush=True,
    )
    print(f"[align] joints={JOINTS}", flush=True)
    print(f"[align] default_q={DEFAULT_Q.tolist()}", flush=True)
    return actor, mean, std, obs_dim, src


class DianDogIsaacMuJoCo:
    """Isaac-faithful closed-loop controller."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, ckpt: Path):
        self.model = model
        self.data = data
        self.actor, self.obs_mean, self.obs_std, self.obs_dim, _ = load_actor(ckpt)
        self.use_scan = self.obs_dim >= PROP_DIM + SCAN_DIM
        self.command = np.zeros(3, dtype=np.float64)
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.scan_xy = make_scan_grid() if self.use_scan else None
        self.spawn_xy = np.array([float(data.qpos[0]), float(data.qpos[1])], dtype=np.float64)

        joint_to_act = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
        qpos_adr, qvel_adr, act_ids = [], [], []
        for name in JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"missing joint {name}")
            qpos_adr.append(int(model.jnt_qposadr[jid]))
            qvel_adr.append(int(model.jnt_dofadr[jid]))
            if jid not in joint_to_act:
                raise RuntimeError(f"no actuator for {name}")
            act_ids.append(joint_to_act[jid])
        self.qpos_adr = np.asarray(qpos_adr)
        self.qvel_adr = np.asarray(qvel_adr)
        self.act_ids = np.asarray(act_ids)
        self.foot_body = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in FEET]
        if any(b < 0 for b in self.foot_body):
            raise RuntimeError(f"missing feet: {self.foot_body}")
        self.base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        mode = f"{self.obs_dim}D" + ("+scan" if self.use_scan else "-blind")
        print(f"[obs] mode={mode}  (Isaac prop=52D)", flush=True)

    def reset(self) -> None:
        self.last_action[:] = 0.0
        self.command[:] = 0.0
        mujoco.mj_resetData(self.model, self.data)
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, int(kid))
        else:
            self.data.qpos[0:3] = [self.spawn_xy[0], self.spawn_xy[1], 0.48]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_adr] = DEFAULT_Q
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # Settle under PD holding default (action=0) — same as Isaac reset hold
        for _ in range(100):
            self.apply_actuators()
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.last_action[:] = 0.0
        self.apply_actuators()

    def foot_contacts(self) -> np.ndarray:
        """Binary contact ≈ Isaac (||F|| > 1) for play; force API omitted for speed."""
        out = np.zeros(4, dtype=np.float32)
        foot_set = {int(b): i for i, b in enumerate(self.foot_body)}
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            for gid in (c.geom1, c.geom2):
                bid = int(self.model.geom_bodyid[gid])
                if bid in foot_set:
                    out[foot_set[bid]] = 1.0
        return out

    def height_scan(self) -> np.ndarray:
        """187D clip(base_z - hit_z - 0.5, -1, 1) — matches go2_env height_data."""
        quat = self.data.qpos[3:7].astype(np.float64)
        yaw = yaw_from_quat(quat)
        c, s = math.cos(yaw), math.sin(yaw)
        bx, by, bz = float(self.data.qpos[0]), float(self.data.qpos[1]), float(self.data.qpos[2])
        ray_z = bz + SCAN_Z_OFFSET
        out = np.empty(SCAN_DIM, dtype=np.float32)
        geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        for i, (lx, ly) in enumerate(self.scan_xy):
            wx = bx + c * lx - s * ly
            wy = by + s * lx + c * ly
            pnt = np.array([wx, wy, ray_z], dtype=np.float64)
            geomid = np.array([-1], dtype=np.int32)
            dist = mujoco.mj_ray(
                self.model, self.data, pnt, vec, geomgroup, 1, int(self.base_body), geomid
            )
            hit_z = bz - 1.0 if dist < 0 or not np.isfinite(dist) else ray_z - float(dist)
            out[i] = float(np.clip(bz - hit_z - 0.5, -1.0, 1.0))
        return out

    def build_obs(self) -> np.ndarray:
        q = self.data.qpos[self.qpos_adr].astype(np.float64)
        dq = self.data.qvel[self.qvel_adr].astype(np.float64)
        quat = self.data.qpos[3:7].astype(np.float64)
        # freejoint: lin vel world, ang vel body — match Isaac root_*_b
        lin_b = quat_rotate_inverse(quat, self.data.qvel[0:3].astype(np.float64))
        ang_b = self.data.qvel[3:6].astype(np.float64)
        grav = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        prop = np.concatenate(
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
        obs = np.concatenate([prop, self.height_scan()]) if self.use_scan else prop
        if obs.shape[0] != self.obs_mean.shape[0]:
            raise RuntimeError(f"obs {obs.shape} != normalizer {self.obs_mean.shape}")
        return ((obs - self.obs_mean) / (self.obs_std + OBS_NORM_EPS)).astype(np.float32)

    def apply_actuators(self) -> None:
        # Isaac: processed = scale * action + default; set_joint_position_target → PD
        tgt = DEFAULT_Q + ACTION_SCALE * self.last_action.astype(np.float64)
        q = self.data.qpos[self.qpos_adr]
        dq = self.data.qvel[self.qvel_adr]
        tau = KP * (tgt - q) - KD * dq
        for i, aid in enumerate(self.act_ids):
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = float(np.clip(tau[i], lo, hi))

    def policy_step(self) -> None:
        """Run policy; when cmd≈0, blend action→0 to suppress 碎步 until RL relearns stand."""
        cmd_n = float(np.linalg.norm(self.command))
        obs = self.build_obs()
        with torch.inference_mode():
            action = self.actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        action = np.clip(action, -100.0, 100.0).astype(np.float32)
        if cmd_n < 0.08:
            # Soft hold-default: decay residual gait (play-side; training also penalizes this)
            self.last_action = (0.85 * self.last_action + 0.15 * action * 0.25).astype(np.float32)
            if float(np.linalg.norm(self.last_action)) < 0.05:
                self.last_action[:] = 0.0
        else:
            self.last_action = action
        self.apply_actuators()

    def status(self) -> str:
        q = self.data.qpos[self.qpos_adr]
        return (
            f"cmd=({self.command[0]:+.2f},{self.command[1]:+.2f},{self.command[2]:+.2f}) "
            f"xyz=({self.data.qpos[0]:.2f},{self.data.qpos[1]:.2f},{self.data.qpos[2]:.3f}) "
            f"|feet={self.foot_contacts().sum():.0f} "
            f"haa=({q[0]:+.2f},{q[1]:+.2f},{q[2]:+.2f},{q[3]:+.2f})"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Isaac-aligned DianDog MuJoCo play")
    ap.add_argument("--scene", type=Path, default=ROOT / "diandog_stairs.xml")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--auto-fwd", action="store_true")
    ap.add_argument("--vx", type=float, default=0.70)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene.expanduser().resolve()))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    ctrl = DianDogIsaacMuJoCo(model, data, args.ckpt.expanduser().resolve())
    ctrl.reset()
    if args.auto_fwd:
        ctrl.command[0] = float(args.vx)

    print(f"[play] settled {ctrl.status()}", flush=True)
    print("[play] 8/2=vx  7/9=vy  4/6=yaw  0/5=stop  R=reset  P=pause", flush=True)

    paused = {"v": False}

    def on_key(keycode: int) -> None:
        if keycode in (ord("P"), ord("p")):
            paused["v"] = not paused["v"]
            print("paused" if paused["v"] else "running", flush=True)
            return
        if keycode in (ord("R"), ord("r")):
            ctrl.reset()
            if args.auto_fwd:
                ctrl.command[0] = float(args.vx)
            print(f"reset | {ctrl.status()}", flush=True)
            return
        if keycode in KEY_STOP:
            ctrl.command[:] = 0.0
        elif keycode in KEY_DELTA:
            axis, sgn = KEY_DELTA[keycode]
            lo, hi = CMD_LIMITS[axis]
            ctrl.command[axis] = float(np.clip(ctrl.command[axis] + sgn * CMD_STEP, lo, hi))
        else:
            return
        print(ctrl.status(), flush=True)

    phys_i = 0
    t0 = time.time()
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.distance = 8.0
        viewer.cam.elevation = -25.0
        viewer.cam.azimuth = 135.0
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
