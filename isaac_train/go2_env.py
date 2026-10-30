# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import wrap_to_pi

from .go2_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg


class Go2Env(DirectRLEnv):
    cfg: Go2FlatEnvCfg | Go2RoughEnvCfg

    def __init__(self, cfg: Go2FlatEnvCfg | Go2RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._prev_prev_actions = torch.zeros_like(self._previous_actions)
        # Domain randomization buffers for privileged obs
        dof_dim = gym.spaces.flatdim(self.single_action_space)
        self._p_gain_scale = torch.ones(self.num_envs, dof_dim, device=self.device)
        self._d_gain_scale = torch.ones(self.num_envs, dof_dim, device=self.device)
        self._dr_friction = torch.full((self.num_envs, 1), float(self.cfg.terrain.physics_material.static_friction), device=self.device)
        self._dr_mass = torch.zeros(self.num_envs, 1, device=self.device)
        self._dr_com = torch.zeros(self.num_envs, 3, device=self.device)
        self._dr_initialized = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Hip / knee joint indices for dedicated posture rewards (pattern configurable)
        hip_joint_pattern = getattr(self.cfg, "hip_joint_pattern", ".*_hip_joint")
        self._hip_joint_ids, _ = self._robot.find_joints(hip_joint_pattern)
        rear_hip_pattern = getattr(self.cfg, "rear_hip_joint_pattern", "r[lr]_haa_j1")
        self._rear_hip_joint_ids, _ = self._robot.find_joints(rear_hip_pattern)
        thigh_joint_pattern = getattr(self.cfg, "thigh_joint_pattern", ".*_hfe_j2")
        self._thigh_joint_ids, _ = self._robot.find_joints(thigh_joint_pattern)
        knee_joint_pattern = getattr(self.cfg, "knee_joint_pattern", ".*_kfe_j3")
        self._knee_joint_ids, _ = self._robot.find_joints(knee_joint_pattern)
        # L/R haa pairs for mirror penalty (defaults sum≈0 → crooked if sum drifts)
        self._haa_fl_ids, _ = self._robot.find_joints("fl_haa_j1")
        self._haa_fr_ids, _ = self._robot.find_joints("fr_haa_j1")
        self._haa_rl_ids, _ = self._robot.find_joints("rl_haa_j1")
        self._haa_rr_ids, _ = self._robot.find_joints("rr_haa_j1")

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        # Store command history and episode start poses for curriculum progression
        self._episode_commands = torch.zeros_like(self._commands)
        self._episode_start_pos = torch.zeros(self.num_envs, 3, device=self.device) # episode가 끝나는 시점에서의 로봇의 현재 위치와, episode 시작점 사이의 거리를 통해 episode내에서 로봇이 얼만큼 걸었나를 판별 -> 이 실제 이동한 거리값을 기준으로 커리큘럼 승급/강등 여부 판단
        self._episode_max_height = torch.zeros(self.num_envs, device=self.device)

        ## Training/Playing 모드 전환을 유연하게 하려고 추가한 플래그 ##
        ### switches for curriculum updates and command sampling
        self._use_curriculum = getattr(self.cfg, "use_curriculum", True)
        self._command_mode = getattr(self.cfg, "command_mode", "random")
        fixed_command_cfg = torch.tensor(
            getattr(self.cfg, "fixed_command", (1.0, 0.0, 0.0)), device=self.device, dtype=torch.float
        )
        self._fixed_command = fixed_command_cfg.unsqueeze(0)
        self._use_heading_command = getattr(self.cfg, "heading_command", False)
        if self._use_heading_command:
            heading_range = torch.tensor(
                getattr(self.cfg, "command_heading_range", (-torch.pi, torch.pi)),
                device=self.device,
                dtype=torch.float,
            )
            yaw_limits = torch.tensor(
                getattr(self.cfg, "command_yaw_range", (-1.0, 1.0)),
                device=self.device,
                dtype=torch.float,
            )
            self._heading_range = heading_range
            self._yaw_limits = yaw_limits
            self._heading_min = float(heading_range[0].item())
            self._heading_max = float(heading_range[1].item())
            self._yaw_min = float(yaw_limits[0].item())
            self._yaw_max = float(yaw_limits[1].item())
            resample_time = getattr(self.cfg, "heading_resample_time_s", 10.0)
            self._heading_resample_steps = max(1, int(resample_time / self.step_dt))
            self._heading_control_stiffness = getattr(self.cfg, "heading_control_stiffness", 0.5)
            self._rel_heading_envs = getattr(self.cfg, "rel_heading_envs", 1.0)
            self._rel_standing_envs = getattr(self.cfg, "rel_standing_envs", 0.02)
            self._heading_targets = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            self._heading_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self._is_heading_env = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self._standing_env = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._command_log_interval = int(getattr(self.cfg, "command_log_interval", 0))
        self._command_log_env = int(getattr(self.cfg, "command_log_env", 0))
        self._command_log_step = 0
        # Speed curriculum (wheeled_dog command_levels): start slow, grow with tracking
        self._use_command_curriculum = bool(getattr(self.cfg, "use_command_curriculum", False))
        self._cmd_scale = float(getattr(self.cfg, "command_range_init_scale", 1.0))
        self._cmd_scale_final = float(getattr(self.cfg, "command_range_final_scale", 1.0))
        self._cmd_scale_step = float(getattr(self.cfg, "command_range_step", 0.05))
        self._cmd_scale_min = float(getattr(self.cfg, "command_range_init_scale", 1.0))
        self._rel_only_lin_x = float(getattr(self.cfg, "rel_only_lin_x_envs", 0.0))
        self._rel_only_lin_y = float(getattr(self.cfg, "rel_only_lin_y_envs", 0.0))
        self._rel_only_ang_z = float(getattr(self.cfg, "rel_only_ang_z_envs", 0.0))
        self._rel_standing = float(getattr(self.cfg, "rel_standing_envs", 0.0))

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "action_smooth_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "upward",
                "base_height",
                "torques",
                "stop_penalty_lin",
                "stop_penalty_ang",
                "stand_still_vel",
                "stand_still_action",
                "stand_still_air",
                "hip_pos",
                "rear_hip_pos",
                "hip_mirror",
                "thigh_pos",
                "feet_stumble",
                "feet_clearance",
                "feet_foothold",
                "feet_miss",
                "feet_slide",
                "knee_pos",
                "dof_close_to_default",
                "work",
            ]
        }
        # Body name patterns (Go2 defaults; DianDog overrides via cfg)
        base_pat = getattr(self.cfg, "base_body_name", "base")
        feet_pat = getattr(self.cfg, "feet_body_names", ".*_foot")
        thigh_pat = getattr(self.cfg, "thigh_body_names", ".*_thigh")
        self._base_id, _ = self._contact_sensor.find_bodies(base_pat)
        self._feet_ids, _ = self._contact_sensor.find_bodies(feet_pat)
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(thigh_pat)
        self._feet_body_ids, _ = self._robot.find_bodies(feet_pat)
        self._capture_dr_from_sim_once()
        self._apply_friction_dr_once()

    def _capture_dr_from_sim_once(self):
        """Capture DR values from the simulation once after startup events."""
        # mass and com: capture actual values from PhysX after startup randomization
        try:
            masses = self._robot.root_physx_view.get_masses().to(self.device)
            base_mass = masses[:, self._base_id]
            if base_mass.dim() > 2:
                base_mass = base_mass.squeeze(1)
            self._dr_mass[:] = base_mass.view(-1, 1)
            print(
                "[DR] base mass sample (per-env):",
                base_mass.detach().cpu().numpy().tolist()[: min(5, len(base_mass))],
                "total mass sample:",
                masses.sum(dim=1).detach().cpu().numpy().tolist()[: min(5, len(masses))],
            )
        except Exception as e:
            print("mass fallback, error:", e)
            if getattr(self._robot.data, "default_mass", None) is not None:
                self._dr_mass[:] = self._robot.data.default_mass[:, self._base_id].view(-1, 1)

        try:
            coms = self._robot.root_physx_view.get_coms().to(self.device)
            base_com = coms[:, self._base_id, :3]
            if base_com.dim() > 3:
                base_com = base_com.squeeze(1)
            self._dr_com[:] = base_com.view(-1, 3)
            print(
                "[DR] base COM sample (per-env):",
                base_com.detach().cpu().numpy().tolist()[: min(5, len(base_com))],
            )
        except Exception as e:
            print("com fallback, error:", e)
            if getattr(self._robot.data, "body_com_pos_b", None) is not None:
                self._dr_com[:] = self._robot.data.body_com_pos_b[:, self._base_id, :].view(-1, 3)

        # PD gain scales: ratio of current actuator gains to defaults
        if self._robot.actuators:
            # assume uniform actuator on go2; take first actuator gains
            actuator = next(iter(self._robot.actuators.values()))
            # shape (num_envs, dof_per_actuator)
            cur_stiffness = actuator.stiffness
            cur_damping = actuator.damping
            default_stiffness = self._robot.data.default_joint_stiffness[:, actuator.joint_indices]
            default_damping = self._robot.data.default_joint_damping[:, actuator.joint_indices]
            # avoid div by zero
            stiff_scale = torch.ones_like(cur_stiffness)
            damp_scale = torch.ones_like(cur_damping)
            stiff_nonzero = torch.abs(default_stiffness) > 1e-6
            damp_nonzero = torch.abs(default_damping) > 1e-6
            stiff_scale[stiff_nonzero] = cur_stiffness[stiff_nonzero] / default_stiffness[stiff_nonzero]
            damp_scale[damp_nonzero] = cur_damping[damp_nonzero] / default_damping[damp_nonzero]
            # broadcast into full dof order
            self._p_gain_scale[:, actuator.joint_indices] = stiff_scale
            self._d_gain_scale[:, actuator.joint_indices] = damp_scale

        # friction: keep as configured terrain friction (events are startup and not easily readable)
        if hasattr(self.cfg.terrain, "physics_material"):
            self._dr_friction[:] = float(self.cfg.terrain.physics_material.static_friction)
        # print("mass sample mean", self._dr_mass.mean())

    def _apply_friction_dr_once(self):
        """Sample friction buckets and apply to robot shapes; mirror into _dr_friction."""
        # parameters
        static_range = (0.7, 1.4)
        dynamic_range = (0.7, 1.4)
        restitution_range = (0.0, 0.0)
        num_buckets = 64

        # sample buckets (CPU as PhysX API expects CPU tensors)
        import torch

        device_cpu = torch.device("cpu")
        buckets = torch.empty((num_buckets, 3), device=device_cpu)
        buckets[:, 0] = torch.rand(num_buckets, device=device_cpu) * (static_range[1] - static_range[0]) + static_range[0]
        buckets[:, 1] = torch.rand(num_buckets, device=device_cpu) * (dynamic_range[1] - dynamic_range[0]) + dynamic_range[0]
        buckets[:, 2] = torch.rand(num_buckets, device=device_cpu) * (restitution_range[1] - restitution_range[0]) + restitution_range[0]

        # PhysX view and shape count
        view = self._robot.root_physx_view
        num_shapes = view.max_shapes
        # current materials shape: (num_envs, num_shapes, 3)
        materials = view.get_material_properties()
        bucket_ids = torch.randint(0, num_buckets, (self.num_envs, num_shapes), device=device_cpu)
        material_samples = buckets[bucket_ids]  # (num_envs, num_shapes, 3)
        materials[:] = material_samples
        indices = torch.arange(num_shapes, device=device_cpu, dtype=torch.int32)
        view.set_material_properties(materials, indices)

        # store one representative friction per env for priv_obs (mean of assigned static friction)
        static_friction = material_samples[:, :, 0]
        self._dr_friction[:] = static_friction.mean(dim=1, keepdim=True).to(self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, Go2RoughEnvCfg):
            # we add a height scanner for perceptive locomotion --> scan dot data
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._update_heading_command()
        if not self._use_heading_command:
            self._maybe_log_commands()
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _update_heading_command(self):
        if not self._use_heading_command:
            return

        # advance resampling timer
        self._heading_timer += 1

        # resample heading targets for active heading environments
        resample_mask = self._heading_timer >= self._heading_resample_steps
        resample_mask &= self._is_heading_env
        resample_mask &= ~self._standing_env
        if torch.any(resample_mask):
            env_ids = torch.nonzero(resample_mask, as_tuple=False).squeeze(-1)
            self._heading_targets[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
                self._heading_min, self._heading_max
            )
            self._heading_timer[env_ids] = 0

        heading_mask = self._is_heading_env & (~self._standing_env)
        if torch.any(heading_mask):
            env_ids = torch.nonzero(heading_mask, as_tuple=False).squeeze(-1)
            heading_error = wrap_to_pi(self._heading_targets[env_ids] - self._robot.data.heading_w[env_ids])
            yaw_command = torch.clamp(
                self._heading_control_stiffness * heading_error, min=self._yaw_min, max=self._yaw_max
            )
            self._commands[env_ids, 2] = yaw_command

        if torch.any(self._standing_env):
            env_ids = torch.nonzero(self._standing_env, as_tuple=False).squeeze(-1)
            self._commands[env_ids] = 0.0
        self._maybe_log_commands()

    def _maybe_log_commands(self):
        if self._command_log_interval <= 0:
            return
        self._command_log_step += 1
        if self._command_log_step % self._command_log_interval != 0:
            return
        env_id = self._command_log_env % self.num_envs
        cmd = self._commands[env_id]
        actual_lin = self._robot.data.root_lin_vel_b[env_id]
        actual_yaw = self._robot.data.root_ang_vel_b[env_id, 2]
        heading = self._robot.data.heading_w[env_id] if hasattr(self._robot.data, "heading_w") else torch.tensor(0.0)
        print(
            "[Go2Env] env"
            f" {env_id}: cmd=({cmd[0].item():+.2f}, {cmd[1].item():+.2f}, {cmd[2].item():+.2f})"
            f" | vel=({actual_lin[0].item():+.2f}, {actual_lin[1].item():+.2f})"
            f" yaw={actual_yaw.item():+.2f} heading={heading.item():+.2f}"
        )

    def _get_observations(self) -> dict:
        self._prev_prev_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()

        height_data = None
        if isinstance(self.cfg, Go2RoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        scan_data = height_data
        policy_scan = scan_data if self.cfg.use_scan_in_policy else None
        critic_scan = scan_data if self.cfg.use_scan_in_critic else None
        scan_first_policy = getattr(self.cfg, "scan_first_in_policy", False)
        scan_first_critic = getattr(self.cfg, "scan_first_in_critic", False)

        # Foot contact flags: 1.0 when contact force on a foot exceeds the threshold.
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        foot_contacts = (
            torch.norm(net_contact_forces[:, -1, self._feet_ids], dim=-1) > 1.0
        ).float()

        prop_obs = torch.cat(
            [
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # 12D
                self._robot.data.joint_vel,  # 12D
                self._robot.data.projected_gravity_b,  # 3D
                self._robot.data.root_lin_vel_b,  # 3D
                self._robot.data.root_ang_vel_b,  # 3D
                self._commands,  # 3D
                self._actions,  # 12D (last action)
                foot_contacts,  # 4D (foot contact flags)
            ],
            dim=-1,
        )
        # actor obs: prop_obs + (optional) scan
        if policy_scan is not None:
            if scan_first_policy:
                policy_obs = torch.cat([policy_scan, prop_obs], dim=-1)
            else:
                policy_obs = torch.cat([prop_obs, policy_scan], dim=-1)
        else:
            policy_obs = prop_obs
        observations = {"policy": policy_obs}

        # Distillation: teacher sees prop+scan (same 239D layout as scanned PPO teacher)
        if getattr(self.cfg, "provide_teacher_obs", False) and scan_data is not None:
            if scan_first_policy:
                observations["teacher"] = torch.cat([scan_data, prop_obs], dim=-1)
            else:
                observations["teacher"] = torch.cat([prop_obs, scan_data], dim=-1)

        # privileged obs (critic only): mass/com + friction + PD gain scales + optional scan
        # use DR buffers to avoid mismatch with internal PhysX sampling
        mass_com = torch.cat([self._dr_mass, self._dr_com], dim=1)  # 4D
        friction_coeff = self._dr_friction

        priv_obs = torch.cat([mass_com, friction_coeff, self._p_gain_scale, self._d_gain_scale], dim=-1)  # 29D
        if critic_scan is not None:
            if scan_first_critic:
                critic_obs = torch.cat([critic_scan, priv_obs, prop_obs], dim=-1)
            else:
                critic_obs = torch.cat([prop_obs, priv_obs, critic_scan], dim=-1)
        else:
            critic_obs = torch.cat([prop_obs, priv_obs], dim=-1)

        # critic receives scan-first or prop-first based on cfg
        observations["critic"] = critic_obs

        # print("angular velocity x/y: ", self._robot.data.root_ang_vel_b[0,2])

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Peak climb vs episode start (NOT vs origin — standing base is already ~0.43m)
        h = self._robot.data.root_link_pose_w[:, 2] - self._episode_start_pos[:, 2]
        self._episode_max_height = torch.maximum(self._episode_max_height, h)
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # stop error
        lin_vel_norm_sq = torch.sum(torch.square(self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        stop_penalty_lin = torch.exp(-2.0 * lin_vel_norm_sq)
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        stop_penalty_ang = torch.exp(-2.0 * ang_vel_error)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate / smooth (2nd difference, wheeled_dog action_smooth_l2)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        action_smooth = torch.sum(
            torch.square(self._actions - 2.0 * self._previous_actions + self._prev_prev_actions), dim=1
        )
        # signed mechanical work (regen clamped out: only positive work penalized)
        joint_power = torch.sum(self._robot.data.applied_torque * self._robot.data.joint_vel, dim=1)
        work_energy = torch.clamp_min(joint_power, 0.0) * self.step_dt
        # feet air time (target shorter on stairs → encourage lift without needing long flight)
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_target = float(getattr(self.cfg, "feet_air_time_target", 0.5))
        air_delta = last_air_time - air_target
        if getattr(self.cfg, "feet_air_time_positive_only", False):
            air_delta = torch.clamp(air_delta, min=0.0)
        air_time = torch.sum(air_delta * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation + soft upright bonus
        # roll-only: projected_gravity_b[:,1] ≈ roll; leave pitch free for stair climb
        if getattr(self.cfg, "flat_orientation_roll_only", False):
            flat_orientation = torch.square(self._robot.data.projected_gravity_b[:, 1])
        else:
            flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        upward = 1.0 - torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # 직접 구현 한 reward function : base height #################################################################

        # Penalize base height away from target
        # base_height = torch.mean(self.base_pos[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # rew_base_height = torch.square(base_height - self.cfg.rewards.base_height_target)
        
        base_height_world = self._robot.data.root_link_pose_w[:, 2]
        if isinstance(self.cfg, Go2RoughEnvCfg):
            ray_hits = self._height_scanner.data.ray_hits_w[..., 2]
            valid_hits = torch.isfinite(ray_hits)
            valid_counts = valid_hits.sum(dim=1)
            sum_hits = torch.where(valid_hits, ray_hits, torch.zeros_like(ray_hits)).sum(dim=1)
            mean_ground_height = torch.where(
                valid_counts > 0,
                sum_hits / torch.clamp(valid_counts, min=1).to(ray_hits.dtype),
                self._terrain.env_origins[:, 2]
                if getattr(self._terrain, "env_origins", None) is not None
                else torch.zeros_like(base_height_world),
            )
            relative_base_height = base_height_world - mean_ground_height
        else:
            relative_base_height = base_height_world

        base_height_target = getattr(self.cfg, "base_height_target", 0.3)
        # Clamp: falling off mesh / bad ray hits otherwise square-explode episode reward
        height_err = torch.clamp(relative_base_height - base_height_target, min=-0.4, max=0.4)
        rew_base_height = torch.square(height_err)
        z_vel_error = torch.clamp(z_vel_error, max=4.0)
        rew_torque = torch.sum(self._robot.data.applied_torque, dim=1)
        rew_dof_close_to_default = torch.sum(
            torch.square(self._robot.data.joint_pos - self._robot.data.default_joint_pos), dim=1
        )
        # Hip-only deviation penalty (parkour-style)
        rew_hip_pos = torch.sum(
            torch.square(
                self._robot.data.joint_pos[:, self._hip_joint_ids]
                - self._robot.data.default_joint_pos[:, self._hip_joint_ids]
            ),
            dim=1,
        )
        # Extra rear-haa penalty (flat: 后髋往内收)
        rear_hip_scale = float(getattr(self.cfg, "rear_hip_pos_reward_scale", 0.0))
        if rear_hip_scale != 0.0 and len(self._rear_hip_joint_ids) > 0:
            rew_rear_hip_pos = torch.sum(
                torch.square(
                    self._robot.data.joint_pos[:, self._rear_hip_joint_ids]
                    - self._robot.data.default_joint_pos[:, self._rear_hip_joint_ids]
                ),
                dim=1,
            )
        else:
            rew_rear_hip_pos = torch.zeros(self.num_envs, device=self.device)
        # L/R haa mirror: fl+fr≈0, rl+rr≈0 — breaks when one leg tucks / other splay
        hip_mirror_scale = float(getattr(self.cfg, "hip_mirror_reward_scale", 0.0))
        if (
            hip_mirror_scale != 0.0
            and len(self._haa_fl_ids) > 0
            and len(self._haa_fr_ids) > 0
            and len(self._haa_rl_ids) > 0
            and len(self._haa_rr_ids) > 0
        ):
            jp = self._robot.data.joint_pos
            front_sum = jp[:, self._haa_fl_ids[0]] + jp[:, self._haa_fr_ids[0]]
            rear_sum = jp[:, self._haa_rl_ids[0]] + jp[:, self._haa_rr_ids[0]]
            rew_hip_mirror = torch.square(front_sum) + torch.square(rear_sum)
        else:
            rew_hip_mirror = torch.zeros(self.num_envs, device=self.device)
        # Thigh (hfe) near default — crooked look often from asymmetric pitch
        thigh_scale = float(getattr(self.cfg, "thigh_pos_reward_scale", 0.0))
        if thigh_scale != 0.0 and len(self._thigh_joint_ids) > 0:
            rew_thigh_pos = torch.sum(
                torch.square(
                    self._robot.data.joint_pos[:, self._thigh_joint_ids]
                    - self._robot.data.default_joint_pos[:, self._thigh_joint_ids]
                ),
                dim=1,
            )
        else:
            rew_thigh_pos = torch.zeros(self.num_envs, device=self.device)
        # Feet stumble: horizontal force dominates vertical support
        net_contact_forces = self._contact_sensor.data.net_forces_w_history[:, -1, self._feet_ids]
        horiz_force = torch.norm(net_contact_forces[..., :2], dim=2)
        vert_force = torch.abs(net_contact_forces[..., 2])
        stumble = torch.any(horiz_force > (self.cfg.feet_stumble_ratio * vert_force), dim=1).float()
        # Swing clearance + foothold + anti-踩空 + anti-drag (scan privileged for reward only)
        clearance_scale = float(getattr(self.cfg, "feet_clearance_reward_scale", 0.0))
        foothold_scale = float(getattr(self.cfg, "feet_foothold_reward_scale", 0.0))
        miss_scale = float(getattr(self.cfg, "feet_miss_reward_scale", 0.0))
        slide_scale = float(getattr(self.cfg, "feet_slide_reward_scale", 0.0))
        if (
            clearance_scale != 0.0
            or foothold_scale != 0.0
            or miss_scale != 0.0
            or slide_scale != 0.0
        ) and isinstance(self.cfg, Go2RoughEnvCfg):
            foot_z = self._robot.data.body_pos_w[:, self._feet_body_ids, 2]
            ground = mean_ground_height.unsqueeze(1)
            clr = foot_z - ground
            contact_n = torch.norm(net_contact_forces, dim=-1)
            swing = (contact_n <= 1.0).float()
            stance = 1.0 - swing
            moving = (torch.norm(self._commands[:, :2], dim=1) > 0.1).float()
            if clearance_scale != 0.0:
                target_clr = float(getattr(self.cfg, "feet_clearance_target", 0.08))
                clr_err = torch.clamp(clr - target_clr, min=-0.15, max=0.15)
                rew_feet_clearance = torch.sum(torch.exp(-torch.square(clr_err) / 0.04) * swing, dim=1) * moving
            else:
                rew_feet_clearance = torch.zeros(self.num_envs, device=self.device)
            # Stance foothold: vertical support dominates + foot near local ground
            if foothold_scale != 0.0:
                safe_v = torch.clamp(vert_force, min=1e-3)
                support_ratio = torch.clamp(safe_v / (horiz_force + safe_v), 0.0, 1.0)
                near_ground = torch.exp(-torch.square(torch.clamp(clr, -0.12, 0.12)) / 0.02)
                rew_feet_foothold = torch.sum(support_ratio * near_ground * stance, dim=1) * moving
            else:
                rew_feet_foothold = torch.zeros(self.num_envs, device=self.device)
            # 踩空: airborne foot dips below local terrain (+margin) → stepped into void/edge
            if miss_scale != 0.0:
                margin = float(getattr(self.cfg, "feet_miss_margin", 0.03))
                miss = ((clr < margin) & (swing > 0.5)).float()
                rew_feet_miss = torch.sum(miss, dim=1) * moving
            else:
                rew_feet_miss = torch.zeros(self.num_envs, device=self.device)
            # 拖地: stance foot still has large horizontal speed (stretched trailing leg)
            if slide_scale != 0.0:
                foot_vxy = self._robot.data.body_lin_vel_w[:, self._feet_body_ids, :2]
                slide = torch.sum(torch.sum(torch.square(foot_vxy), dim=-1) * stance, dim=1)
                rew_feet_slide = slide * moving
            else:
                rew_feet_slide = torch.zeros(self.num_envs, device=self.device)
        else:
            rew_feet_clearance = torch.zeros(self.num_envs, device=self.device)
            rew_feet_foothold = torch.zeros(self.num_envs, device=self.device)
            rew_feet_miss = torch.zeros(self.num_envs, device=self.device)
            rew_feet_slide = torch.zeros(self.num_envs, device=self.device)
        # Knee stretch: keep kfe near default crouch (anti 伸直拖后腿)
        knee_scale = float(getattr(self.cfg, "knee_pos_reward_scale", 0.0))
        if knee_scale != 0.0 and len(self._knee_joint_ids) > 0:
            rew_knee_pos = torch.sum(
                torch.square(
                    self._robot.data.joint_pos[:, self._knee_joint_ids]
                    - self._robot.data.default_joint_pos[:, self._knee_joint_ids]
                ),
                dim=1,
            )
        else:
            rew_knee_pos = torch.zeros(self.num_envs, device=self.device)
        # Zero-cmd stand still: kill 原地碎步 / 漂移 (gate on |cmd|)
        stand_vel_scale = float(getattr(self.cfg, "stand_still_vel_reward_scale", 0.0))
        stand_act_scale = float(getattr(self.cfg, "stand_still_action_reward_scale", 0.0))
        stand_air_scale = float(getattr(self.cfg, "stand_still_air_reward_scale", 0.0))
        stand_cmd_thr = float(getattr(self.cfg, "stand_still_cmd_threshold", 0.1))
        cmd_small = (
            (torch.norm(self._commands[:, :2], dim=1) < stand_cmd_thr)
            & (torch.abs(self._commands[:, 2]) < stand_cmd_thr)
        ).float()
        if stand_vel_scale != 0.0:
            yaw_rate_sq = torch.square(self._robot.data.root_ang_vel_b[:, 2])
            rew_stand_vel = cmd_small * (lin_vel_norm_sq + yaw_rate_sq)
        else:
            rew_stand_vel = torch.zeros(self.num_envs, device=self.device)
        if stand_act_scale != 0.0:
            rew_stand_action = cmd_small * torch.sum(torch.square(self._actions), dim=1)
        else:
            rew_stand_action = torch.zeros(self.num_envs, device=self.device)
        if stand_air_scale != 0.0:
            # net_contact_forces already (N, n_feet, 3) from stumble block
            foot_air = (torch.norm(net_contact_forces, dim=-1) <= 1.0).float()
            rew_stand_air = cmd_small * torch.sum(foot_air, dim=1)
        else:
            rew_stand_air = torch.zeros(self.num_envs, device=self.device)
        ##########################################################################################################

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "action_smooth_l2": action_smooth
            * float(getattr(self.cfg, "action_smooth_reward_scale", 0.0))
            * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "upward": upward * float(getattr(self.cfg, "upward_reward_scale", 0.0)) * self.step_dt,
            "base_height": rew_base_height * self.cfg.base_height_reward_scale * self.step_dt,
            "torques": rew_torque * self.cfg.torque_reward_scale * self.step_dt,
            "stop_penalty_lin": stop_penalty_lin * self.cfg.stop_penalty_reward_scale * self.step_dt,
            "stop_penalty_ang": stop_penalty_ang * self.cfg.stop_penalty_reward_scale * self.step_dt,
            "stand_still_vel": rew_stand_vel * stand_vel_scale * self.step_dt,
            "stand_still_action": rew_stand_action * stand_act_scale * self.step_dt,
            "stand_still_air": rew_stand_air * stand_air_scale * self.step_dt,
            "hip_pos": rew_hip_pos * self.cfg.hip_pos_reward_scale * self.step_dt,
            "rear_hip_pos": rew_rear_hip_pos * rear_hip_scale * self.step_dt,
            "hip_mirror": rew_hip_mirror * hip_mirror_scale * self.step_dt,
            "thigh_pos": rew_thigh_pos * thigh_scale * self.step_dt,
            "feet_stumble": stumble * self.cfg.feet_stumble_reward_scale * self.step_dt,
            "feet_clearance": rew_feet_clearance * clearance_scale * self.step_dt,
            "feet_foothold": rew_feet_foothold * foothold_scale * self.step_dt,
            "feet_miss": rew_feet_miss * miss_scale * self.step_dt,
            "feet_slide": rew_feet_slide * slide_scale * self.step_dt,
            "knee_pos": rew_knee_pos * knee_scale * self.step_dt,
            "dof_close_to_default": (
                rew_dof_close_to_default * self.cfg.dof_close_to_default_reward_scale * self.step_dt
            ),
            "work": work_energy * self.cfg.work_reward_scale,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        curriculum_log: dict[str, float] = dict()
        if (
            isinstance(self.cfg, Go2RoughEnvCfg)
            and getattr(self._terrain, "terrain_origins", None) is not None
            and self._use_curriculum
        ):
            prev_mask = self.episode_length_buf[env_ids] > 0 # episode_length_buf는 0이면 막 리셋돼서 아직 한 발도 떼지 않은 에피소드 시작점 -> buffer updates at "(episode_length_buf += 1) at source/isaaclab/isaaclab/envs/direct_rl_env.py:368."
            if torch.any(prev_mask): # prev_mask에 최소한 한 요소라도 0보다 크면
                prev_env_ids = env_ids[prev_mask]
                current_pos = self._robot.data.root_link_pose_w[prev_env_ids, :3]
                # Isaac Lab / Legged Gym: distance from terrain env origin (pit center), not episode start
                origins = self._terrain.env_origins[prev_env_ids]
                distance = torch.norm(current_pos[:, :2] - origins[:, :2], dim=1)
                start_pos = self._episode_start_pos[prev_env_ids]
                # Height vs episode start (standing on flat ≈ 0; crest ≈ N*step_h)
                height_gain = current_pos[:, 2] - start_pos[:, 2]
                height_vs_origin = current_pos[:, 2] - origins[:, 2]
                forward = current_pos[:, 0] - start_pos[:, 0]
                # Progress along +X from terrain origin (strip stairs face +X)
                x_from_origin = current_pos[:, 0] - origins[:, 0]
                progress = torch.maximum(distance, forward)
                terrain_length = self.cfg.terrain.terrain_generator.size[0]
                move_up_frac = float(getattr(self.cfg, "curriculum_move_up_fraction", 0.5))
                move_down_frac = float(getattr(self.cfg, "curriculum_move_down_fraction", 0.5))

                # Column layout follows sub_terrain proportions (curriculum generator).
                num_cols = int(self.cfg.terrain.terrain_generator.num_cols)
                stairs_prop = float(getattr(self.cfg, "curriculum_stairs_proportion", 0.0))
                inv_prop = float(getattr(self.cfg, "curriculum_inv_stairs_proportion", 0.0))
                flat_prop = float(getattr(self.cfg, "curriculum_flat_proportion", 0.0))
                n_stairs_cols = int(round(stairs_prop * num_cols)) if stairs_prop > 0 else 0
                n_inv_cols = int(round(inv_prop * num_cols)) if inv_prop > 0 else 0
                n_flat_cols = int(round(flat_prop * num_cols)) if flat_prop > 0 else 0
                terrain_cols = self._terrain.terrain_types[prev_env_ids]
                if n_stairs_cols > 0:
                    is_stairs = terrain_cols < n_stairs_cols
                elif n_flat_cols > 0:
                    is_stairs = terrain_cols >= n_flat_cols
                else:
                    is_stairs = torch.ones_like(terrain_cols, dtype=torch.bool)
                is_inv = terrain_cols < n_inv_cols if n_inv_cols > 0 else torch.zeros_like(terrain_cols, dtype=torch.bool)
                is_non_stairs = ~is_stairs

                stairs_only = bool(getattr(self.cfg, "curriculum_stairs_only", False))
                promote_x = float(getattr(self.cfg, "curriculum_stairs_promote_x", 0.0))
                min_h = float(getattr(self.cfg, "curriculum_stairs_min_height_gain", 0.0))
                # Strip: past first crest + actually climbed (peak vs start). Optional back-down.
                min_peak = float(getattr(self.cfg, "curriculum_stairs_min_peak_height", 0.0))
                max_end_h = float(getattr(self.cfg, "curriculum_stairs_max_end_height", 1e9))
                peak_h = self._episode_max_height[prev_env_ids]
                if promote_x > 0.0:
                    move_up = x_from_origin > promote_x
                else:
                    # Standard: > half terrain tile ≈ walked to outer rim / out of pit
                    move_up = distance > (terrain_length * move_up_frac)
                if min_peak > 0.0 and torch.any(is_stairs & ~is_inv):
                    strip = is_stairs & ~is_inv
                    # _episode_max_height is ALREADY relative to episode start (see _get_rewards)
                    cleared_crest = peak_h > min_peak
                    back_down = height_gain < max_end_h
                    move_up = torch.where(strip, move_up & cleared_crest & back_down, move_up)
                # Inv pits: climb-out vs terrain origin (pit floor)
                if min_h > 0.0:
                    climbed = height_vs_origin > min_h
                    h_mask = is_inv if getattr(self.cfg, "curriculum_height_gain_inv_only", True) else is_stairs
                    if torch.any(h_mask):
                        move_up = torch.where(h_mask, move_up & climbed, move_up)
                if stairs_only:
                    move_up = move_up & is_stairs

                if getattr(self.cfg, "curriculum_disable_demote", False):
                    move_down = torch.zeros_like(move_up)
                elif getattr(self.cfg, "curriculum_demote_use_command", False):
                    # Isaac/wheeled: distance < |cmd_xy| * episode_s * fraction
                    cmd_xy = torch.norm(self._episode_commands[prev_env_ids, :2], dim=1)
                    expected = cmd_xy * self.max_episode_length_s * move_down_frac
                    move_down = (distance < expected) & (~move_up)
                else:
                    almost_stuck = progress < (terrain_length * move_down_frac)
                    timed_out = self.reset_time_outs[prev_env_ids]
                    move_down = almost_stuck & (~move_up) & (timed_out | (progress < 0.25))
                if stairs_only:
                    move_down = move_down & is_stairs

                # Avoid Isaac default: promote-at-max → random lower level (mean keeps falling)
                if getattr(self.cfg, "curriculum_clamp_max_level", False):
                    levels = self._terrain.terrain_levels[prev_env_ids].to(dtype=torch.long)
                    levels = levels + move_up.long() - move_down.long()
                    max_lvl = int(self._terrain.max_terrain_level) - 1
                    levels = torch.clamp(levels, min=0, max=max(0, max_lvl))
                    # Non-stairs keep level (don't inflate / dilute stairs curriculum)
                    if stairs_only and torch.any(is_non_stairs):
                        levels = torch.where(
                            is_non_stairs,
                            self._terrain.terrain_levels[prev_env_ids].to(dtype=torch.long),
                            levels,
                        )
                    self._terrain.terrain_levels[prev_env_ids] = levels
                    self._terrain.env_origins[prev_env_ids] = self._terrain.terrain_origins[
                        levels, self._terrain.terrain_types[prev_env_ids]
                    ]
                else:
                    self._terrain.update_env_origins(prev_env_ids, move_up, move_down)

                all_levels = self._terrain.terrain_levels.float()
                curriculum_log["Curriculum/mean_terrain_level"] = torch.mean(all_levels).item()
                if n_stairs_cols > 0:
                    stairs_mask = self._terrain.terrain_types < n_stairs_cols
                    if torch.any(stairs_mask):
                        curriculum_log["Curriculum/stairs_mean_level"] = torch.mean(
                            all_levels[stairs_mask]
                        ).item()
                    curriculum_log["Curriculum/stairs_promote_rate"] = (
                        float(move_up.float().mean().item()) if move_up.numel() else 0.0
                    )

        elif isinstance(self.cfg, Go2RoughEnvCfg) and getattr(self._terrain, "terrain_origins", None) is not None:
            # Curriculum off: randomize both terrain level (row) and type (column) per reset for broader coverage.
            new_levels = torch.randint(
                0,
                self._terrain.max_terrain_level,
                (len(env_ids),),
                device=self.device,
            )
            num_types = self._terrain.terrain_origins.shape[1]
            new_types = torch.randint(0, num_types, (len(env_ids),), device=self.device)
            self._terrain.terrain_levels[env_ids] = new_levels
            self._terrain.terrain_types[env_ids] = new_types
            self._terrain.env_origins[env_ids] = self._terrain.terrain_origins[new_levels, new_types]

            # # Debugging: 샘플링된 지형 레벨과 타입 확인
            # preview = min(5, len(env_ids)) # 로그를 출력할 때 다 보이면 너무 많으니, 최대 5개만 보여줌.
            # print(
            #     "[Go2Env] Curriculum disabled reset → sampled levels",
            #     new_levels[:preview].detach().cpu().tolist(),
            #     "types",
            #     new_types[:preview].detach().cpu().tolist(),
            # )

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._prev_prev_actions[env_ids] = 0.0
        # Sample new commands
        # 여기에서 command 처리할때 원래 크기는 envs, 3이라서 값이 이 샘플링을 통해 다 바뀌어버림.
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(1.0, 1.0)

        # 각각 random sampling 구현 ###########################################################################
        # Domain randomization: sampled buffers for privileged obs
        num_resets = len(env_ids)

        # Speed curriculum: raise when tracking good; pull back when tracking collapses / terrain drops
        if self._use_command_curriculum and num_resets > 0:
            track = self._episode_sums["track_lin_vel_xy_exp"][env_ids]
            max_track = abs(float(self.cfg.lin_vel_reward_scale))
            min_terrain = float(getattr(self.cfg, "command_curriculum_min_terrain_level", 0.0))
            mean_terrain = 0.0
            if getattr(self._terrain, "terrain_levels", None) is not None:
                mean_terrain = float(torch.mean(self._terrain.terrain_levels.float()).item())
            if max_track > 1e-6:
                mean_frac = float(torch.mean(track).item()) / max(self.max_episode_length_s, 1e-6) / max_track
                if (
                    mean_frac > 0.80
                    and mean_terrain >= min_terrain
                    and self._cmd_scale < self._cmd_scale_final - 1e-6
                ):
                    self._cmd_scale = min(self._cmd_scale + self._cmd_scale_step, self._cmd_scale_final)
                elif mean_frac < 0.35 and self._cmd_scale > self._cmd_scale_min + 1e-6:
                    self._cmd_scale = max(self._cmd_scale - self._cmd_scale_step, self._cmd_scale_min)
                curriculum_log["Curriculum/command_scale"] = self._cmd_scale

        ## Training/Playing 모드 전환을 유연하게 하려고 추가한 플래그 ##
        ### Training Mode: 디폴트는 random sampling - random command 숫자 바꾸려면 여기 수정
        ### Playing Mode: 디폴트는 fixed command
        if self._command_mode == "random":
            cmd_scale = self._cmd_scale if self._use_command_curriculum else 1.0
            lin_x_range = getattr(self.cfg, "command_lin_x_range", (-1.0, 1.0))
            lin_y_range = getattr(self.cfg, "command_lin_y_range", (-1.0, 1.0))
            ang_vel_range = getattr(self.cfg, "command_yaw_range", (-1.0, 1.0))
            x0, x1 = float(lin_x_range[0]) * cmd_scale, float(lin_x_range[1]) * cmd_scale
            y0, y1 = float(lin_y_range[0]) * cmd_scale, float(lin_y_range[1]) * cmd_scale
            w0, w1 = float(ang_vel_range[0]) * cmd_scale, float(ang_vel_range[1]) * cmd_scale

            rand_x = torch.rand(num_resets, 1, device=self.device) * (x1 - x0) + x0
            rand_y = torch.rand(num_resets, 1, device=self.device) * (y1 - y0) + y0
            rand_yaw = torch.rand(num_resets, 1, device=self.device) * (w1 - w0) + w0
            # Bias flat omni toward usable speeds: 35% resample |vx| into outer band
            # (fixes slow shuffle + under-trained reverse from uniform mid-band mass).
            if num_resets > 0 and float(getattr(self.cfg, "command_min_lin_speed", 0.0)) > 0.0:
                hi = max(abs(x0), abs(x1))
                lo_band = min(0.55 * hi, max(0.45, 0.55 * hi))
                boost = torch.rand(num_resets, 1, device=self.device) < 0.35
                if torch.any(boost):
                    sign = torch.where(
                        torch.rand(int(boost.sum().item()), 1, device=self.device) < 0.5, -1.0, 1.0
                    )
                    mag = torch.rand(int(boost.sum().item()), 1, device=self.device) * (hi - lo_band) + lo_band
                    rand_x = rand_x.clone()
                    rand_x[boost.squeeze(1)] = sign * mag
            self._commands[env_ids] = torch.cat([rand_x, rand_y, rand_yaw], dim=1)

            # Omni mixture buckets (wheeled_dog rel_only_*)
            if num_resets > 0:
                r = torch.rand(num_resets, device=self.device)
                c0 = self._rel_standing
                c1 = c0 + self._rel_only_lin_x
                c2 = c1 + self._rel_only_lin_y
                c3 = c2 + self._rel_only_ang_z
                stand = r < c0
                only_x = (r >= c0) & (r < c1)
                only_y = (r >= c1) & (r < c2)
                only_yaw = (r >= c2) & (r < c3)
                if torch.any(stand):
                    self._commands[env_ids[stand]] = 0.0
                if torch.any(only_x):
                    self._commands[env_ids[only_x], 1:] = 0.0
                if torch.any(only_y):
                    self._commands[env_ids[only_y], 0] = 0.0
                    self._commands[env_ids[only_y], 2] = 0.0
                if torch.any(only_yaw):
                    self._commands[env_ids[only_yaw], :2] = 0.0

            min_lin = float(getattr(self.cfg, "command_min_lin_speed", 0.0))
            if min_lin > 0.0 and num_resets > 0:
                xy = self._commands[env_ids, :2]
                spd = torch.norm(xy, dim=1, keepdim=True).clamp(min=1e-6)
                need = spd.squeeze(1) < min_lin
                # don't inflate standing / pure-yaw buckets
                need &= ~stand if num_resets > 0 else need
                need &= ~only_yaw if num_resets > 0 else need
                if torch.any(need):
                    scale = (min_lin / spd[need]).clamp(max=3.0)
                    self._commands[env_ids[need], :2] = xy[need] * scale
        elif self._command_mode == "fixed":
            self._commands[env_ids] = self._fixed_command.expand(num_resets, -1)
        else:
            raise ValueError(f"Unsupported command mode: {self._command_mode}")

        # Stairs envs: force forward-only; flat keeps omni sampling above
        if (
            getattr(self.cfg, "command_stairs_forward_only", False)
            and num_resets > 0
            and getattr(self, "_terrain", None) is not None
            and getattr(self._terrain, "terrain_types", None) is not None
        ):
            stairs_prop = float(getattr(self.cfg, "curriculum_stairs_proportion", 0.0))
            num_cols = int(self.cfg.terrain.terrain_generator.num_cols)
            n_stairs = int(round(stairs_prop * num_cols)) if stairs_prop > 0 else 0
            if n_stairs > 0:
                is_stairs = self._terrain.terrain_types[env_ids] < n_stairs
                if torch.any(is_stairs):
                    self._commands[env_ids[is_stairs]] = self._fixed_command.expand(
                        int(is_stairs.sum().item()), -1
                    )

        if self._use_heading_command:
            self._heading_timer[env_ids] = 0
            self._standing_env[env_ids] = False
            self._is_heading_env[env_ids] = False

            if num_resets > 0:
                stand_mask = torch.rand(num_resets, device=self.device) < self._rel_standing_envs
                if torch.any(stand_mask):
                    stand_ids = env_ids[stand_mask]
                    self._standing_env[stand_ids] = True
                    self._commands[stand_ids] = 0.0

                active_ids = env_ids[~stand_mask] if torch.any(stand_mask) else env_ids
                if len(active_ids) > 0:
                    heading_mask = torch.rand(len(active_ids), device=self.device) < self._rel_heading_envs
                    if torch.any(heading_mask):
                        heading_ids = active_ids[heading_mask]
                        self._is_heading_env[heading_ids] = True
                        self._heading_targets[heading_ids] = torch.empty(len(heading_ids), device=self.device).uniform_(
                            self._heading_min, self._heading_max
                        )
                        self._commands[heading_ids, 2] = 0.0

        self._episode_commands[env_ids] = self._commands[env_ids]


        # ## THIS IS FOR PLAYING
        # '''
        # # 고정된 x방향 속도로만 이동 (2.5 m/s)
        # fixed_x = torch.ones(num_resets, 1, device=self.device) * 2.5  # x방향 2.5 m/s
        # fixed_y = torch.zeros(num_resets, 1, device=self.device)       # y방향 0.0 m/s
        # fixed_yaw = torch.zeros(num_resets, 1, device=self.device)     # yaw 0.0 rad/s
        # self._commands[env_ids] = torch.cat([fixed_x, fixed_y, fixed_yaw], dim=1)   
        # '''
        # # 턴하면서 빠르게 이동
        # fixed_x = torch.ones(num_resets, 1,
        # device=self.device) * 1.0   # x방향 1.0 m/s
        # fixed_y = torch.zeros(num_resets, 1,
        # device=self.device)        # y방향 0.0 m/s
        # fixed_yaw = torch.ones(num_resets, 1,
        # device=self.device) * 0.0  # yaw 0.0 rad/s
        # self._commands[env_ids] = torch.cat([fixed_x,
        # fixed_y, fixed_yaw], dim=1)        
        # self._episode_commands[env_ids] = self._commands[env_ids] # 지금 막 샘플링한 속도 명령(self._commands[env_ids])을 그대로 _episode_commands 버퍼에 복사해 두는 역할. 이렇게 저장해 둬야 에피소드가 끝날 때 “이 환경은 어떤 속도를 명령받았었나?”를 알 수 있고, 그 값을 이용해 기대 이동 거리(expected_distance)를 계산해서 커리큘럼 강등 여부를 판단
        ######################################################################################################
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._episode_start_pos[env_ids] = default_root_state[:, :3] # 이렇게 초기 좌표를 기록해 둬야 에피소드가 끝날 때 current_pos - start_pos로 실제 이동 거리를 계산할 수 있음.
        self._episode_max_height[env_ids] = 0.0

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
        if curriculum_log:
            self.extras["log"].update(curriculum_log)
