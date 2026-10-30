# 训练与评测笔记（DianDog-BlindStairs-v1.0）

## 环境依赖（Isaac 训练）

- OS：Ubuntu 22.04  
- GPU：NVIDIA（本机 RTX 5080 验证）  
- Conda：`isaaclab51`  
- 工程：`extreme-quadruped-parkour` + Isaac Lab 2.x  
- 并行环境数：常用 `NUM_ENVS=4096`  
- 任务：`DianDog-Stairs-Omni-Student-RL-v0`

## 关键脚本环境变量

```bash
GUI=1
NUM_ENVS=4096
MAX_ITERS=10000
STUDENT_CHECKPOINT=/path/to/distill_or_prev.pt
# 或 RESUME=1 LOAD_RUN=... CHECKPOINT=model_xxx.pt
```

`scripts/train_diandog_stairs_student_rl.sh` 会：

1. 激活 conda + Isaac Sim 环境  
2. 设置 `PYTHONPATH` 指向 parkour 的 `isaaclab_tasks` 等  
3. 若未 `RESUME`，导出 `STUDENT_CHECKPOINT` 供 train.py 只加载 actor  

## 蒸馏 → 微调注意点

- 蒸馏 ckpt 含 `student_state_dict`；PPO 微调后为 `actor_state_dict`。  
- Critic 维度在「是否 scan-in-critic」切换时不兼容，微调时常 **仅加载 actor**、critic 重初始化。  
- `obs_normalizer` 必须与权重一并加载，否则动作分布偏移导致站不稳。  

## 课程踩坑记录

1. `min_peak` 若误用绝对高度（含机身 ~0.43m），会几乎恒满足而瞬间顶满课程。  
2. `_episode_max_height` 已是相对起点爬升，课程侧勿再减 `start_z`。  
3. 课程顶满后若继续猛加爬楼奖励，易破坏平地步态与后髋姿态。  

## 站立碎步问题

现象：`cmd=0` 时仍小碎步并缓慢漂移。  

原因：训练里站立样本不足；空气时间/跟踪奖励在低速仍驱动迈步。  

v1.0 对策：

- 提高 `rel_standing_envs`  
- 增加 `stand_still_vel / action / air` 门控惩罚  
- MuJoCo play 在 `|cmd|<0.08` 时衰减动作至默认  

## MuJoCo 迁移

| 项 | Isaac | MuJoCo 本包 |
|----|-------|-------------|
| 引擎 | PhysX | MuJoCo |
| 控制 | set_joint_position_target + DelayedPD | 显式 PD 力矩 motor |
| 楼梯 | MeshStairsStrip | `gen_stairs_xml.py` 盒体 + 倒角 |
| 网格 | USD | STL |

建议评测顺序：零指令 → 平地前进 → 15cm → 20cm。

## 硬件对照

`docs/dianzu.txt` 记录关节侧电阻等参数，便于实机驱动标定；仿真 PD 与实机电流环增益需另行辨识，不可直接等同。

## 推荐引用文件

发布评测报告时请注明：

- 权重：`checkpoints/policy_blind52_iter1200.pt`  
- 视频：`media/demo_mujoco_blind_climb.mp4`  
- 配置：`configs/train_env_snapshot.yaml`  
