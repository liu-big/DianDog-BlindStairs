# 奖励与环境实现参考（源码摘录索引）
本文档索引发布包内 `isaac_train/go2_env.py` / `go2_env_cfg.py` 的关键符号，便于审阅与刷库说明。

## EnvCfg 类列表
- `class EventCfg:`
- `class Go2FlatEnvCfg(DirectRLEnvCfg):`
- `class Go2RoughEnvCfg(Go2FlatEnvCfg):`
- `class Go2RoughPlayEnvCfg(Go2RoughEnvCfg):`
- `class Go2RoughAbl1EnvCfg(Go2RoughEnvCfg):`
- `class Go2RoughAbl1PlayEnvCfg(Go2RoughPlayEnvCfg):`
- `class Go2RoughAbl2_5EnvCfg(Go2RoughEnvCfg):`
- `class Go2RoughAbl2_5PlayEnvCfg(Go2RoughPlayEnvCfg):`
- `class DianDogEventCfg:`
- `class DianDogFlatEnvCfg(Go2FlatEnvCfg):`
- `class DianDogRoughEnvCfg(Go2RoughEnvCfg):`
- `class DianDogRoughPlayEnvCfg(DianDogRoughEnvCfg):`
- `class DianDogStairsEnvCfg(DianDogRoughEnvCfg):`
- `class DianDogStairsPlayEnvCfg(DianDogStairsEnvCfg):`
- `class DianDogStairsOmniEnvCfg(DianDogRoughEnvCfg):`
- `class DianDogStairsOmniPlayEnvCfg(DianDogStairsOmniEnvCfg):`
- `class DianDogStairsOmniStudentEnvCfg(DianDogStairsOmniEnvCfg):`
- `class DianDogStairsOmniStudentPlayEnvCfg(DianDogStairsOmniStudentEnvCfg):`
- `class DianDogStairsOmniStudentFinetuneEnvCfg(DianDogStairsOmniEnvCfg):`
- `class DianDogStairsOmniStudentFinetunePlayEnvCfg(DianDogStairsOmniStudentFinetuneEnvCfg):`

## StudentFinetune 关键字段（文本检索）

### `stand_still`

```
 165|     stand_still_vel_reward_scale: float = 0.0
 166|     stand_still_action_reward_scale: float = 0.0
 167|     stand_still_air_reward_scale: float = 0.0
 168|     stand_still_cmd_threshold: float = 0.1
 745|     stand_still_vel_reward_scale = -3.5
 746|     stand_still_action_reward_scale = -1.5
 747|     stand_still_air_reward_scale = -1.2
 748|     stand_still_cmd_threshold: float = 0.10
```

### `hip_mirror`

```
 161|     hip_mirror_reward_scale: float = 0.0  # L+R haa should stay ~0
 730|     hip_mirror_reward_scale = -5.0
```

### `rear_hip`

```
 159|     rear_hip_pos_reward_scale: float = 0.0  # extra rear-haa penalty (0=off)
 160|     rear_hip_joint_pattern: str = "r[lr]_haa_j1"
 728|     rear_hip_pos_reward_scale = -6.0
 729|     rear_hip_joint_pattern: str = "r[lr]_haa_j1"
```

### `feet_clearance`

```
 172|     feet_clearance_reward_scale: float = 0.0
 173|     feet_clearance_target: float = 0.08
 717|     feet_clearance_reward_scale = 0.40
 718|     feet_clearance_target: float = 0.06
```

### `feet_foothold`

```
 175|     feet_foothold_reward_scale: float = 0.0
 719|     feet_foothold_reward_scale = 1.1
```

### `feet_miss`

```
 176|     feet_miss_reward_scale: float = 0.0  # negative recommended
 177|     feet_miss_margin: float = 0.03  # foot below local ground+margin while airborne → 踩空
 720|     feet_miss_reward_scale = -2.0
 721|     feet_miss_margin: float = 0.03
```

### `curriculum_stairs`

```
 590|     curriculum_stairs_promote_x: float = 6.0
 591|     curriculum_stairs_min_peak_height: float = 0.14
 592|     curriculum_stairs_max_end_height: float = 1.0
 593|     curriculum_stairs_only: bool = True
 596|     curriculum_stairs_proportion: float = 0.5
 598|     curriculum_stairs_min_height_gain: float = 0.0
 751|     curriculum_stairs_promote_x: float = 7.0
 752|     curriculum_stairs_min_peak_height: float = 0.12
 756|     curriculum_stairs_proportion: float = 0.45
```

### `step_height_range`

```
 286|                     step_height_range=(0.05, 0.23),
 294|                     step_height_range=(0.1, 0.45),
 515|                     step_height_range=(0.04, 0.18),
 642|                     step_height_range=(0.02, 0.20),
 773|                     step_height_range=(0.02, 0.20),  # target max 20cm
```

### `use_scan_in_policy`

```
  86|     use_scan_in_policy: bool = False
 230|     use_scan_in_policy: bool = True
 358|         self.observation_space = self.num_prop_obs + (self.num_scan_obs if self.use_scan_in_policy else 0)
 381|     use_scan_in_policy: bool = False
 386|     use_scan_in_policy: bool = False
 558|     use_scan_in_policy: bool = True
 685|     use_scan_in_policy: bool = False
 709|     use_scan_in_policy: bool = False
```

### `action_scale`

```
  79|     action_scale = 0.25
 454|     action_scale = 0.3
 469|     action_scale = 0.3
 582|     action_scale = 0.3
```

### `rel_standing`

```
 199|     rel_standing_envs: float = 0.02
 353|     rel_standing_envs: float = 0.0
 576|     rel_standing_envs: float = 0.02
 744|     rel_standing_envs: float = 0.28
```

## go2_env.py 关键方法
- L23: `def __init__(self, cfg: Go2FlatEnvCfg | Go2RoughEnvCfg, render_mode: str | None = None, **kwargs):`
- L160: `def _capture_dr_from_sim_once(self):`
- L220: `def _apply_friction_dr_once(self):`
- L252: `def _setup_scene(self):`
- L273: `def _pre_physics_step(self, actions: torch.Tensor):`
- L280: `def _apply_action(self):`
- L283: `def _update_heading_command(self):`
- L315: `def _maybe_log_commands(self):`
- L333: `def _get_observations(self) -> dict:`
- L405: `def _get_rewards(self) -> torch.Tensor:`
- L675: `def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:`
- L681: `def _reset_idx(self, env_ids: torch.Tensor | None):`

## _get_rewards 分段注释
- L406: # Peak climb vs episode start (NOT vs origin — standing base is already ~0.43m)
- L409: # linear velocity tracking
- L412: # yaw rate tracking
- L415: # stop error
- L418: # angular velocity x/y
- L421: # z velocity tracking
- L423: # joint torques
- L425: # joint acceleration
- L427: # action rate / smooth (2nd difference, wheeled_dog action_smooth_l2)
- L432: # signed mechanical work (regen clamped out: only positive work penalized)
- L435: # feet air time (target shorter on stairs → encourage lift without needing long flight)
- L445: # undesired contacts
- L451: # flat orientation + soft upright bonus
- L452: # roll-only: projected_gravity_b[:,1] ≈ roll; leave pitch free for stair climb
- L459: # 직접 구현 한 reward function : base height #################################################################
- L461: # Penalize base height away from target
- L462: # base_height = torch.mean(self.base_pos[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
- L463: # rew_base_height = torch.square(base_height - self.cfg.rewards.base_height_target)
- L483: # Clamp: falling off mesh / bad ray hits otherwise square-explode episode reward
- L485: rew_base_height = torch.square(height_err)
- L487: rew_torque = torch.sum(self._robot.data.applied_torque, dim=1)
- L488: rew_dof_close_to_default = torch.sum(
- L491: # Hip-only deviation penalty (parkour-style)
- L492: rew_hip_pos = torch.sum(
- L499: # Extra rear-haa penalty (flat: 后髋往内收)
- L502: rew_rear_hip_pos = torch.sum(
- L510: rew_rear_hip_pos = torch.zeros(self.num_envs, device=self.device)
- L511: # L/R haa mirror: fl+fr≈0, rl+rr≈0 — breaks when one leg tucks / other splay
- L512: hip_mirror_scale = float(getattr(self.cfg, "hip_mirror_reward_scale", 0.0))
- L514: hip_mirror_scale != 0.0
- L523: rew_hip_mirror = torch.square(front_sum) + torch.square(rear_sum)
- L525: rew_hip_mirror = torch.zeros(self.num_envs, device=self.device)
- L526: # Thigh (hfe) near default — crooked look often from asymmetric pitch
- L529: rew_thigh_pos = torch.sum(
- L537: rew_thigh_pos = torch.zeros(self.num_envs, device=self.device)
- L538: # Feet stumble: horizontal force dominates vertical support
- L543: # Swing clearance + foothold + anti-踩空 + anti-drag (scan privileged for reward only)
- L564: rew_feet_clearance = torch.sum(torch.exp(-torch.square(clr_err) / 0.04) * swing, dim=1) * moving
- L566: rew_feet_clearance = torch.zeros(self.num_envs, device=self.device)
- L567: # Stance foothold: vertical support dominates + foot near local ground
- L572: rew_feet_foothold = torch.sum(support_ratio * near_ground * stance, dim=1) * moving
- L574: rew_feet_foothold = torch.zeros(self.num_envs, device=self.device)
- L575: # 踩空: airborne foot dips below local terrain (+margin) → stepped into void/edge
- L579: rew_feet_miss = torch.sum(miss, dim=1) * moving
- L581: rew_feet_miss = torch.zeros(self.num_envs, device=self.device)
- L582: # 拖地: stance foot still has large horizontal speed (stretched trailing leg)
- L586: rew_feet_slide = slide * moving
- L588: rew_feet_slide = torch.zeros(self.num_envs, device=self.device)
- L590: rew_feet_clearance = torch.zeros(self.num_envs, device=self.device)
- L591: rew_feet_foothold = torch.zeros(self.num_envs, device=self.device)
- L592: rew_feet_miss = torch.zeros(self.num_envs, device=self.device)
- L593: rew_feet_slide = torch.zeros(self.num_envs, device=self.device)
- L594: # Knee stretch: keep kfe near default crouch (anti 伸直拖后腿)
- L597: rew_knee_pos = torch.sum(
- L605: rew_knee_pos = torch.zeros(self.num_envs, device=self.device)
- L606: # Zero-cmd stand still: kill 原地碎步 / 漂移 (gate on |cmd|)
- L607: stand_vel_scale = float(getattr(self.cfg, "stand_still_vel_reward_scale", 0.0))
- L608: stand_act_scale = float(getattr(self.cfg, "stand_still_action_reward_scale", 0.0))
- L609: stand_air_scale = float(getattr(self.cfg, "stand_still_air_reward_scale", 0.0))
- L610: stand_cmd_thr = float(getattr(self.cfg, "stand_still_cmd_threshold", 0.1))
- L617: rew_stand_vel = cmd_small * (lin_vel_norm_sq + yaw_rate_sq)
- L619: rew_stand_vel = torch.zeros(self.num_envs, device=self.device)
- L621: rew_stand_action = cmd_small * torch.sum(torch.square(self._actions), dim=1)
- L623: rew_stand_action = torch.zeros(self.num_envs, device=self.device)
- L625: # net_contact_forces already (N, n_feet, 3) from stumble block
- L627: rew_stand_air = cmd_small * torch.sum(foot_air, dim=1)
- L629: rew_stand_air = torch.zeros(self.num_envs, device=self.device)
- L630: ##########################################################################################################
- L632: rewards = {
- L647: "base_height": rew_base_height * self.cfg.base_height_reward_scale * self.step_dt,
- L648: "torques": rew_torque * self.cfg.torque_reward_scale * self.step_dt,
- L651: "stand_still_vel": rew_stand_vel * stand_vel_scale * self.step_dt,
- L652: "stand_still_action": rew_stand_action * stand_act_scale * self.step_dt,
- L653: "stand_still_air": rew_stand_air * stand_air_scale * self.step_dt,
- L654: "hip_pos": rew_hip_pos * self.cfg.hip_pos_reward_scale * self.step_dt,
- L655: "rear_hip_pos": rew_rear_hip_pos * rear_hip_scale * self.step_dt,
- L656: "hip_mirror": rew_hip_mirror * hip_mirror_scale * self.step_dt,
- L657: "thigh_pos": rew_thigh_pos * thigh_scale * self.step_dt,
- L659: "feet_clearance": rew_feet_clearance * clearance_scale * self.step_dt,
- L660: "feet_foothold": rew_feet_foothold * foothold_scale * self.step_dt,
- L661: "feet_miss": rew_feet_miss * miss_scale * self.step_dt,
- L662: "feet_slide": rew_feet_slide * slide_scale * self.step_dt,
- L663: "knee_pos": rew_knee_pos * knee_scale * self.step_dt,
- L665: rew_dof_close_to_default * self.cfg.dof_close_to_default_reward_scale * self.step_dt
- L670: # Logging
