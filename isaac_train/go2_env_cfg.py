# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import (
    MeshDebrisTerrainCfg,
    MeshGapStripTerrainCfg,
    MeshHurdleStripTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshParkourStepTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshStairsStripTerrainCfg,
)
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
from isaaclab_assets.robots.diandog import DIANDOG_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for randomization."""

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
        },
    )

    randomize_pd = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "distribution": "uniform",
            "operation": "scale",
        },
    )


@configclass
class Go2FlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = 12
    observation_space = 52
    state_space = 81  # 52 prop + 29 priv
    num_prop_obs: int = 52
    num_priv_obs: int = 29
    num_scan_obs: int = 0
    use_scan_in_policy: bool = False
    use_scan_in_critic: bool = False
    scan_first_in_policy: bool = False
    scan_first_in_critic: bool = False
    # Distillation: emit extra "teacher" obs (prop+scan) while policy stays prop-only
    provide_teacher_obs: bool = False

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    # Enable self-collisions and bump solver iterations so limbs don't pass through each other.
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UNITREE_GO2_CFG.spawn.replace(
            articulation_props=UNITREE_GO2_CFG.spawn.articulation_props.replace(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            )
        ),
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # reward scales
    lin_vel_reward_scale = 3.0
    yaw_rate_reward_scale = 1.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 2.0
    undesired_contact_reward_scale = -0.0
    flat_orientation_reward_scale = -0.5
    base_height_reward_scale = -10.0
    torque_reward_scale = 0.0
    stop_penalty_reward_scale = 0.0
    dof_close_to_default_reward_scale = -0.05
    work_reward_scale = -0.01
    hip_pos_reward_scale = -0.5  # penalty on hip joints drifting from defaults
    hip_joint_pattern: str = ".*_hip_joint"
    rear_hip_pos_reward_scale: float = 0.0  # extra rear-haa penalty (0=off)
    rear_hip_joint_pattern: str = "r[lr]_haa_j1"
    hip_mirror_reward_scale: float = 0.0  # L+R haa should stay ~0
    thigh_pos_reward_scale: float = 0.0
    thigh_joint_pattern: str = ".*_hfe_j2"
    # Zero-command stand still (negative scales): kill 碎步 / drift
    stand_still_vel_reward_scale: float = 0.0
    stand_still_action_reward_scale: float = 0.0
    stand_still_air_reward_scale: float = 0.0
    stand_still_cmd_threshold: float = 0.1
    feet_stumble_reward_scale = -1.0
    feet_stumble_ratio = 4.0  # horizontal force threshold factor vs vertical
    # Swing-foot clearance vs terrain (privileged scan for reward only; 0=off)
    feet_clearance_reward_scale: float = 0.0
    feet_clearance_target: float = 0.08
    # Stance quality / anti miss-step (privileged; 0=off)
    feet_foothold_reward_scale: float = 0.0
    feet_miss_reward_scale: float = 0.0  # negative recommended
    feet_miss_margin: float = 0.03  # foot below local ground+margin while airborne → 踩空
    # Stance-foot drag / stretched knee (negative recommended for slide)
    feet_slide_reward_scale: float = 0.0
    knee_pos_reward_scale: float = 0.0
    knee_joint_pattern: str = ".*_kfe_j3"
    # wheeled_dog-style extras (0 = off for flat/default)
    action_smooth_reward_scale = 0.0
    upward_reward_scale = 0.0
    # If True: only penalize roll (gy), leave pitch free for stairs
    flat_orientation_roll_only: bool = False
    # If True: feet_air_time only rewards (no penalty when air < target)
    feet_air_time_positive_only: bool = False

    # Training/Playing 모드 전환을 유연하게 하려고 추가한 플래그 
    ## command / curriculum settings (overridden by rough configs as needed)
    command_mode: str = "random"  # "random" or "fixed"
    fixed_command: tuple[float, float, float] = (1.0, 0.0, 0.0)
    use_curriculum: bool = True
    heading_command: bool = False
    heading_resample_time_s: float = 10.0
    heading_control_stiffness: float = 0.5
    rel_heading_envs: float = 1.0
    rel_standing_envs: float = 0.02
    # Omni mixture (wheeled_dog): exclusive buckets; remainder = full (vx,vy,yaw)
    rel_only_lin_x_envs: float = 0.0
    rel_only_lin_y_envs: float = 0.0
    rel_only_ang_z_envs: float = 0.0
    command_lin_x_range: tuple[float, float] = (-1.0, 1.0)
    command_lin_y_range: tuple[float, float] = (-1.0, 1.0)
    command_heading_range: tuple[float, float] = (-math.pi, math.pi)
    command_yaw_range: tuple[float, float] = (-1.0, 1.0)
    command_min_lin_speed: float = 0.0
    # Speed curriculum: sample within scale * final ranges (start low → tip less)
    use_command_curriculum: bool = False
    command_range_init_scale: float = 0.35
    command_range_final_scale: float = 1.0
    command_range_step: float = 0.05
    command_log_interval: int = 0
    command_log_env: int = 0


@configclass
class Go2RoughEnvCfg(Go2FlatEnvCfg):
    # gap terrain constants (used for curriculum + spawn offset)
    gap_subterrain_key: str = "gap_bar"
    hurdle_subterrain_key: str = "hurdle_strip"
    gap_spawn_offset: float = -1.0  # spawn further back on gap/hurdle tiles to give more run-up

    # env
    # policy: 52 prop + 187 scan = 239; critic: 52 prop + 29 priv + 187 scan = 268
    observation_space = 239
    state_space = 268
    num_scan_obs: int = 187
    use_scan_in_policy: bool = True
    use_scan_in_critic: bool = True

    sim: SimulationCfg = Go2FlatEnvCfg().sim.replace(
        physx=Go2FlatEnvCfg().sim.physx.replace(gpu_max_rigid_patch_count=12 * 2**15)
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(
            size=(23.0, 6.0),  # Terrain Size 23m X 6m -> x축으로만 직진하니까!
            num_rows=10,  # level 0~9 단계까지
            num_cols=18,  # gap/hurdle/stairs/parkour each occupy 3 columns, others 2 columns
            sub_terrains={
                # 18컬럼에 7타입 배치 (대체로 균등 비율)
                "boxes": ROUGH_TERRAINS_CFG.sub_terrains["boxes"].replace(
                    proportion=(2 / 18), grid_height_range=(0.025, 0.1)
                ),
                "random_rough": ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].replace(
                    proportion=(2 / 18), noise_range=(0.01, 0.06), noise_step=0.01
                ),
                "debris_field": MeshDebrisTerrainCfg(
                    proportion=(2 / 18),
                    size=(23.0, 23.0),
                    num_debris_min=20,
                    num_debris_max=40,
                    ground_thickness=0.1,
                    box_length_range=(0.5, 2.0),
                    box_width_range=(0.2, 0.6),
                    box_thickness_range=(0.05, 0.25),
                    cyl_radius_range=(0.05, 0.2),
                    cyl_length_range=(0.5, 2.0),
                ),
                "gap_bar": MeshGapStripTerrainCfg(
                    proportion=(3 / 18),
                    size=(23.0, 23.0),
                    gap_width_range=(0.1, 0.8),
                    landing_length=0.45,
                    start_platform_length=8.0,  # longer run-up for the gap strip
                ),
                # hurdle strip: run-up then repeated hurdles with height/gap increasing in difficulty
                "hurdle_strip": MeshHurdleStripTerrainCfg(
                    proportion=(3 / 18),
                    size=(23.0, 23.0),
                    hurdle_height_range=(0.05, 0.3),
                    hurdle_thickness=0.2,
                    hurdle_gap_range=(0.7, 2.0),
                    start_platform_length=3.0,
                ),
                # stairs strip: run-up then up/down stair segments
                "stairs_strip": MeshStairsStripTerrainCfg(
                    proportion=(3 / 18),
                    size=(23.0, 23.0),
                    start_platform_length=3.0,
                    segment_length=5.0,
                    step_height_range=(0.05, 0.23),
                    steps_per_segment=10,
                    pattern=("up", "down", "up", "down"),
                ),
                "parkour_step": MeshParkourStepTerrainCfg(
                    proportion=(3 / 18),
                    size=(23.0, 23.0),
                    start_platform_length=3.0,
                    step_height_range=(0.1, 0.45),
                    step_length_base_range=(0.3, 1.5),
                    steps=6,
                ),
            },
        ),
        max_init_terrain_level=5, # 사수님이 9였다가 1로 바꾸심 -> 내가 5로 바꿈
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Test25 reward scales (override from flat config)
    base_height_reward_scale = 0.0 # Test4 (considering managerbased curriculum)
    flat_orientation_reward_scale = 0.0 # 험지니까 몸이 엄청 기울거라서
    feet_air_time_reward_scale = 0.125 # Test25 (@ Extreme Parkour: 0)
    lin_vel_reward_scale = 3.0 # Test23
    yaw_rate_reward_scale = 1.5 # Test23
    z_vel_reward_scale = -0.0 # Test9 (jumping 많은 지형 고려) # 참고: Extreme Parkour: -0.5
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5 # 참고: Extreme Parkour: -1.0e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01 # Test 22(Hmm didn't overcome Gap so didn't consider "Extreme Parkour: -0.1 scale")
    undesired_contact_reward_scale = -0.8 # Test4 (considering managerbased curriculum)
    torque_reward_scale = 0.0
    stop_penalty_reward_scale = 0.0
    dof_close_to_default_reward_scale = -0.01 # Test24 ((Hmm didn't overcome Gap so didn't consider "Extreme Parkour: -0.04")
    work_reward_scale = -0.003 # Test 12
    hip_pos_reward_scale = -0.3  # Test24 (@ Extreme Parkour: -0.5)
    feet_stumble_reward_scale = -0.0 # Test23 (@ Extreme Parkour: -1.0)
    feet_stumble_ratio = 4.0 # Reward Scale이 아니라, lateral force > fleet_stumble_ratio*|vertical force| 이면 패널티 주도록(옆으로 미끌리는거 방지)

    # keep curriculum active and random commands for training
    command_mode: str = "fixed"
    use_curriculum: bool = True
    heading_command: bool = True
    # heading settings (parkour-like, but keep 10s resample + fixed 1 m/s command)
    command_heading_range: tuple[float, float] = (0.0, 0.0)
    heading_control_stiffness: float = 0.8
    rel_heading_envs: float = 1.0
    rel_standing_envs: float = 0.0
    # keep yaw clamp explicit for clarity
    command_yaw_range: tuple[float, float] = (-1.0, 1.0)

    def __post_init__(self):
        self.observation_space = self.num_prop_obs + (self.num_scan_obs if self.use_scan_in_policy else 0)
        self.state_space = self.num_prop_obs + self.num_priv_obs + (self.num_scan_obs if self.use_scan_in_critic else 0)


@configclass
class Go2RoughPlayEnvCfg(Go2RoughEnvCfg):
    """Evaluation configuration for Go2 rough terrain."""

    # disable curriculum updates so difficulty stays fixed per reset
    use_curriculum: bool = False

    # run with fixed velocity commands during playbacks
    command_mode: str = "fixed"
    fixed_command: tuple[float, float, float] = (1.0, 0.0, 0.0)

    # generate a different random rough terrain by changing the seed and disabling curriculum in generator
    terrain = Go2RoughEnvCfg().terrain.replace(
        terrain_generator=ROUGH_TERRAINS_CFG.replace(curriculum=False, seed=424242)
    )


@configclass
class Go2RoughAbl1EnvCfg(Go2RoughEnvCfg):
    use_scan_in_policy: bool = False


@configclass
class Go2RoughAbl1PlayEnvCfg(Go2RoughPlayEnvCfg):
    use_scan_in_policy: bool = False


@configclass
class Go2RoughAbl2_5EnvCfg(Go2RoughEnvCfg):
    scan_first_in_policy: bool = True
    scan_first_in_critic: bool = True


@configclass
class Go2RoughAbl2_5PlayEnvCfg(Go2RoughPlayEnvCfg):
    scan_first_in_policy: bool = True
    scan_first_in_critic: bool = True


# ---------------------------------------------------------------------------
# DianDog (~18 kg point-foot) — same DirectRL / Ablation stack as Go2
# ---------------------------------------------------------------------------


@configclass
class DianDogEventCfg:
    """DR events for DianDog (base body is ``base_link``)."""

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
        },
    )

    randomize_pd = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "distribution": "uniform",
            "operation": "scale",
        },
    )


@configclass
class DianDogFlatEnvCfg(Go2FlatEnvCfg):
    """Flat warmup for DianDog (same obs/reward layout as Go2 Direct)."""

    events: DianDogEventCfg = DianDogEventCfg()
    robot: ArticulationCfg = DIANDOG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    hip_joint_pattern: str = ".*_haa_j1"
    base_body_name: str = "base_link"
    feet_body_names: str = ".*_ft"
    # Undesired: hip/thigh/shank (base uses termination, not this penalty)
    thigh_body_names: str = ".*(haa|hfe|kfe)"
    base_height_target: float = 0.43
    action_scale = 0.3
    undesired_contact_reward_scale = -1.0


@configclass
class DianDogRoughEnvCfg(Go2RoughEnvCfg):
    """Parkour rough terrain for DianDog (Abl 3.5 default env)."""

    events: DianDogEventCfg = DianDogEventCfg()
    robot: ArticulationCfg = DIANDOG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    hip_joint_pattern: str = ".*_haa_j1"
    base_body_name: str = "base_link"
    feet_body_names: str = ".*_ft"
    thigh_body_names: str = ".*(haa|hfe|kfe)"
    base_height_target: float = 0.43
    action_scale = 0.3
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class DianDogRoughPlayEnvCfg(DianDogRoughEnvCfg):
    use_curriculum: bool = False
    command_mode: str = "fixed"
    fixed_command: tuple[float, float, float] = (1.0, 0.0, 0.0)
    terrain = DianDogRoughEnvCfg().terrain.replace(
        terrain_generator=DianDogRoughEnvCfg().terrain.terrain_generator.replace(
            curriculum=False, seed=424242
        )
    )


@configclass
class DianDogStairsEnvCfg(DianDogRoughEnvCfg):
    """Legacy Abl3.5 stairs (239D scan) — prefer DianDogStairsOmniEnvCfg."""

    episode_length_s = 25.0
    command_mode: str = "fixed"
    fixed_command: tuple[float, float, float] = (1.0, 0.0, 0.0)
    use_curriculum: bool = True

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(
            size=(30.0, 6.0),
            num_rows=10,
            num_cols=12,
            curriculum=True,
            sub_terrains={
                "stairs_climb": MeshStairsStripTerrainCfg(
                    proportion=1.0,
                    size=(30.0, 30.0),
                    start_platform_length=2.5,
                    segment_length=5.5,
                    step_height_range=(0.04, 0.18),
                    steps_per_segment=12,
                    pattern=("up", "up", "up", "down", "down"),
                ),
            },
        ),
        max_init_terrain_level=3,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )


@configclass
class DianDogStairsPlayEnvCfg(DianDogStairsEnvCfg):
    use_curriculum: bool = False
    command_mode: str = "fixed"
    fixed_command: tuple[float, float, float] = (1.0, 0.0, 0.0)
    terrain = DianDogStairsEnvCfg().terrain.replace(
        terrain_generator=DianDogStairsEnvCfg().terrain.terrain_generator.replace(
            curriculum=False, seed=424242
        )
    )


@configclass
class DianDogStairsOmniEnvCfg(DianDogRoughEnvCfg):
    """对齐 extreme-quadruped-parkour：一半平地 + 一半往前爬楼梯 + 高度扫描。

    - 观测：52D prop + 187D height scan（与 Abl3.5 / 网上楼梯方案一致，非盲）
    - 楼梯：条带 2→20 cm；平地全向、楼梯前进
    """

    # Height map in policy+critic (cannot resume flat 52D)
    use_scan_in_policy: bool = True
    use_scan_in_critic: bool = True
    num_scan_obs: int = 187

    episode_length_s = 20.0
    # Flat: omni random; stairs overridden to fixed forward in env
    command_mode: str = "random"
    # Stairs climb faster (teacher solidify → later blind distill)
    fixed_command: tuple[float, float, float] = (1.1, 0.0, 0.0)
    command_stairs_forward_only: bool = True
    heading_command: bool = False
    # Flat omni: higher top speed + min speed
    command_lin_x_range: tuple[float, float] = (-1.2, 1.2)
    command_lin_y_range: tuple[float, float] = (-0.7, 0.7)
    command_yaw_range: tuple[float, float] = (-1.2, 1.2)
    command_min_lin_speed: float = 0.50
    use_curriculum: bool = True
    use_command_curriculum: bool = False
    rel_standing_envs: float = 0.02
    # More pure ±vx (includes reverse) — omni mix alone under-practices backpedal
    rel_only_lin_x_envs: float = 0.22
    rel_only_lin_y_envs: float = 0.08
    rel_only_ang_z_envs: float = 0.05
    rel_heading_envs: float = 0.0
    action_scale = 0.3
    base_height_target: float = 0.43
    curriculum_move_up_fraction: float = 0.5
    curriculum_move_down_fraction: float = 0.25
    curriculum_disable_demote: bool = False
    curriculum_demote_use_command: bool = True
    curriculum_clamp_max_level: bool = True
    # Past first crest (~5.4m); clear summit before promote
    curriculum_stairs_promote_x: float = 6.0
    curriculum_stairs_min_peak_height: float = 0.14
    curriculum_stairs_max_end_height: float = 1.0
    curriculum_stairs_only: bool = True
    curriculum_height_gain_inv_only: bool = True
    curriculum_flat_proportion: float = 0.5
    curriculum_stairs_proportion: float = 0.5
    curriculum_inv_stairs_proportion: float = 0.0
    curriculum_stairs_min_height_gain: float = 0.0
    feet_air_time_target: float = 0.10
    feet_air_time_positive_only: bool = True

    thigh_body_names: str = ".*(hfe|kfe)"
    # Stronger anti-belly/shin jam at crest
    undesired_contact_reward_scale = -1.2
    flat_orientation_roll_only: bool = True
    flat_orientation_reward_scale = -3.0
    upward_reward_scale = 0.25
    base_height_reward_scale = 0.0
    z_vel_reward_scale = -0.25
    ang_vel_reward_scale = -0.05
    lin_vel_reward_scale = 5.5  # track higher commanded speeds
    yaw_rate_reward_scale = 2.8
    feet_air_time_reward_scale = 3.5
    # Penalize hip adduction / abduction drift on flat (haa vs default)
    hip_pos_reward_scale = -0.35
    dof_close_to_default_reward_scale = -0.01
    work_reward_scale = -0.001
    action_rate_reward_scale = -0.012
    action_smooth_reward_scale = -0.018
    joint_torque_reward_scale = -2.0e-5
    joint_accel_reward_scale = -2.5e-7
    feet_stumble_reward_scale = -0.25

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(
            size=(12.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            curriculum=True,
            sub_terrains={
                # 50% stairs strip first → curriculum cols 0..9
                # Start ~2cm (ETH/Isaac: single-step curriculum), ramp to 20cm
                "stairs_strip": MeshStairsStripTerrainCfg(
                    proportion=0.5,
                    size=(12.0, 8.0),
                    start_platform_length=3.0,
                    end_platform_length=2.0,
                    segment_length=2.4,  # 8 steps → tread ≈ 0.30 m
                    step_height_range=(0.02, 0.20),
                    steps_per_segment=8,
                    pattern=("up", "down", "up", "down"),
                    add_side_walls=True,
                ),
                # 50% flat → cols 10..19 (levels frozen when stairs_only)
                "flat": MeshPlaneTerrainCfg(
                    proportion=0.5,
                    size=(12.0, 8.0),
                ),
            },
        ),
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )


@configclass
class DianDogStairsOmniPlayEnvCfg(DianDogStairsOmniEnvCfg):
    use_curriculum: bool = False
    command_mode: str = "random"
    terrain = DianDogStairsOmniEnvCfg().terrain.replace(
        terrain_generator=DianDogStairsOmniEnvCfg().terrain.terrain_generator.replace(
            curriculum=False, seed=424242
        )
    )


@configclass
class DianDogStairsOmniStudentEnvCfg(DianDogStairsOmniEnvCfg):
    """Blind 52D student distill: policy=prop only; teacher obs keeps 239D prop+scan."""

    use_scan_in_policy: bool = False
    use_scan_in_critic: bool = True  # keep scan sensor path / critic layout
    provide_teacher_obs: bool = True


@configclass
class DianDogStairsOmniStudentPlayEnvCfg(DianDogStairsOmniStudentEnvCfg):
    use_curriculum: bool = False
    command_mode: str = "random"
    provide_teacher_obs: bool = False
    terrain = DianDogStairsOmniStudentEnvCfg().terrain.replace(
        terrain_generator=DianDogStairsOmniStudentEnvCfg().terrain.terrain_generator.replace(
            curriculum=False, seed=424242
        )
    )


@configclass
class DianDogStairsOmniStudentFinetuneEnvCfg(DianDogStairsOmniEnvCfg):
    """Blind 52D PPO finetune after distill: active foot lift, anti shin-jam.

    Policy stays prop-only; height scan is reward/critic privileged only.
    """

    use_scan_in_policy: bool = False
    use_scan_in_critic: bool = True
    provide_teacher_obs: bool = False
    # Posture-first finetune from distill: don't let climb rewards destroy flat gait
    thigh_body_names: str = ".*(hfe|kfe)"
    undesired_contact_reward_scale = -1.8
    feet_stumble_reward_scale = -1.0
    # Climb assist (privileged) — modest so posture/stand still can dominate flat
    feet_clearance_reward_scale = 0.40
    feet_clearance_target: float = 0.06
    feet_foothold_reward_scale = 1.1
    feet_miss_reward_scale = -2.0
    feet_miss_margin: float = 0.03
    feet_slide_reward_scale = -0.55
    feet_air_time_target: float = 0.09
    feet_air_time_reward_scale = 2.0
    feet_air_time_positive_only: bool = True
    # Posture: anti 歪腿
    hip_pos_reward_scale = -3.0
    rear_hip_pos_reward_scale = -6.0
    rear_hip_joint_pattern: str = "r[lr]_haa_j1"
    hip_mirror_reward_scale = -5.0
    thigh_pos_reward_scale = -1.5
    thigh_joint_pattern: str = ".*_hfe_j2"
    knee_pos_reward_scale = -1.0
    knee_joint_pattern: str = ".*_kfe_j3"
    dof_close_to_default_reward_scale = -0.25
    base_height_reward_scale = -8.0
    base_height_target: float = 0.43
    upward_reward_scale = 0.45
    flat_orientation_reward_scale = -5.5
    action_rate_reward_scale = -0.020
    action_smooth_reward_scale = -0.028
    lin_vel_reward_scale = 5.0
    # Stand still when cmd≈0: no 碎步 / 漂移
    rel_standing_envs: float = 0.28
    stand_still_vel_reward_scale = -3.5
    stand_still_action_reward_scale = -1.5
    stand_still_air_reward_scale = -1.2
    stand_still_cmd_threshold: float = 0.10
    # Max 20cm; slower promote
    fixed_command: tuple[float, float, float] = (0.85, 0.0, 0.0)
    curriculum_stairs_promote_x: float = 7.0
    curriculum_stairs_min_peak_height: float = 0.12
    curriculum_move_up_fraction: float = 0.75
    curriculum_move_down_fraction: float = 0.50
    curriculum_flat_proportion: float = 0.55
    curriculum_stairs_proportion: float = 0.45
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(
            size=(12.0, 8.0),
            border_width=20.0,
            num_rows=16,  # finer difficulty steps → reach 20cm slower
            num_cols=20,
            curriculum=True,
            sub_terrains={
                "stairs_strip": MeshStairsStripTerrainCfg(
                    proportion=0.45,
                    size=(12.0, 8.0),
                    start_platform_length=3.0,
                    end_platform_length=2.0,
                    segment_length=2.4,
                    step_height_range=(0.02, 0.20),  # target max 20cm
                    steps_per_segment=8,
                    pattern=("up", "down", "up", "down"),
                    add_side_walls=True,
                ),
                "flat": MeshPlaneTerrainCfg(
                    proportion=0.55,
                    size=(12.0, 8.0),
                ),
            },
        ),
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )


@configclass
class DianDogStairsOmniStudentFinetunePlayEnvCfg(DianDogStairsOmniStudentFinetuneEnvCfg):
    use_curriculum: bool = False
    command_mode: str = "random"
    terrain = DianDogStairsOmniStudentFinetuneEnvCfg().terrain.replace(
        terrain_generator=DianDogStairsOmniStudentFinetuneEnvCfg().terrain.terrain_generator.replace(
            curriculum=False, seed=424242
        )
    )
