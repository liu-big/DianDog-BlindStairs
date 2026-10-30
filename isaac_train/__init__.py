# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Go2-Direct-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg",
    },
)

# Playing Mode: Gym에 새로운 환경 ID인 Go2-Rough-Direct-Play-v0를 등록하는 코드
gym.register(
    id="Go2-Rough-Direct-Play-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl1-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughAbl1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl1PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl2_5-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughAbl2_5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl2_5PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl3_5-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl3_5PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl4_0-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl4_0PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl7_0-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl7_0PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl1-Play-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughAbl1PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl1PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl2_5-Play-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughAbl2_5PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl2_5PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl3_5-Play-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl3_5PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl4_0-Play-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl4_0PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl7_0-Play-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl7_0PPORunnerCfg",
    },
)

# DianDog (point-foot) — reuse Go2 DirectRL env + Abl 3.5 policy
gym.register(
    id="DianDog-Direct-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:DianDogFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DianDogFlatPPORunnerCfg",
    },
)

gym.register(
    id="DianDog-Rough-Direct-Abl3_5-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:DianDogRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DianDogRoughAbl3_5PPORunnerCfg",
    },
)

gym.register(
    id="DianDog-Rough-Direct-Abl3_5-Play-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:DianDogRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DianDogRoughAbl3_5PPORunnerCfg",
    },
)

# DianDog stairs: bottom spawn → climb-heavy up/down strip
gym.register(
    id="DianDog-Stairs-Direct-Abl3_5-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:DianDogStairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DianDogStairsAbl3_5PPORunnerCfg",
    },
)

gym.register(
    id="DianDog-Stairs-Direct-Abl3_5-Play-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:DianDogStairsPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DianDogStairsAbl3_5PPORunnerCfg",
    },
)

# Omni stairs finetune from flat (52D, no scan)
gym.register(
    id="DianDog-Stairs-Omni-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:DianDogStairsOmniEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DianDogStairsOmniPPORunnerCfg",
    },
)

gym.register(
    id="DianDog-Stairs-Omni-Play-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:DianDogStairsOmniPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DianDogStairsOmniPPORunnerCfg",
    },
)

# Blind 52D student distill (DAgger) from 239D scanned teacher
gym.register(
    id="DianDog-Stairs-Omni-Student-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:DianDogStairsOmniStudentEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DianDogStairsOmniStudentDistillRunnerCfg"
        ),
    },
)

gym.register(
    id="DianDog-Stairs-Omni-Student-Play-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:DianDogStairsOmniStudentPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DianDogStairsOmniStudentDistillRunnerCfg"
        ),
    },
)

# Blind 52D PPO finetune (active lift / anti shin-jam after distill)
gym.register(
    id="DianDog-Stairs-Omni-Student-RL-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:DianDogStairsOmniStudentFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DianDogStairsOmniStudentPPORunnerCfg"
        ),
    },
)

gym.register(
    id="DianDog-Stairs-Omni-Student-RL-Play-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:DianDogStairsOmniStudentFinetunePlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DianDogStairsOmniStudentPPORunnerCfg"
        ),
    },
)
