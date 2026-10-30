# Copyright (c) 2025 DianDog / Extreme Parkour adaptation
# SPDX-License-Identifier: BSD-3-Clause

"""DianDog (point-foot) for extreme-quadruped-parkour (Isaac Lab 2.2).

USD: ``/home/hi/liu/diandog/usd/diandog.usd`` (from SLDASM + dianzu.txt).
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_DIANDOG_ROOT = Path("/home/hi/liu/diandog")
_USD_CANDIDATES = [
    _DIANDOG_ROOT / "usd" / "diandog.usd",
    _DIANDOG_ROOT / "usd" / "diandog.usda",
]
_USD = next((p for p in _USD_CANDIDATES if p.is_file()), _USD_CANDIDATES[0])

# Must match flat model_4999 defaults (haa/hfe/kfe). Changing this breaks resume.
_STAND = {
    "fl_haa_j1": 0.04,
    "fr_haa_j1": -0.04,
    "rl_haa_j1": -0.04,
    "rr_haa_j1": 0.04,
    "fl_hfe_j2": -0.25,
    "fr_hfe_j2": 0.25,
    "rl_hfe_j2": -0.25,
    "rr_hfe_j2": 0.25,
    "fl_kfe_j3": -0.40,
    "fr_kfe_j3": 0.40,
    "rl_kfe_j3": -0.40,
    "rr_kfe_j3": 0.40,
}

DIANDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_USD),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Off: thigh/shank boxes overlap base at stand pose → instant base_contact.
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos=_STAND,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_haa_j1", ".*_hfe_j2", ".*_kfe_j3"],
            effort_limit=40.0,
            velocity_limit=15.0,
            stiffness=80.0,
            damping=3.5,
            friction=0.0,
            armature=0.01,
            min_delay=0,
            max_delay=1,
        ),
    },
)
"""DianDog point-foot articulation for parkour DirectRL env."""
