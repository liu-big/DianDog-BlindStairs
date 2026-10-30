# 训练脉络与超参说明（DianDog-BlindStairs-v1.0）

本文档对应发布包主权重 `policy_blind52_iter1200.pt`，用于复现实验设置与消融对照。

## 1. 总体流水线

```text
Teacher (239D = 52 prop + 187 height-scan)
        │  DAgger / Distillation
        ▼
Student Blind (52D only)
        │  PPO finetune (scan → reward + critic only)
        ▼
v1.0 Standstill + Posture + Climb  (≈ iter 1200 @ 2026-10-30_19-00-30)
```

### 1.1 Teacher 阶段目标

- 在 Isaac 楼梯条带地形上稳定攀爬、下楼不完全崩坏。  
- 策略可读取 RayCaster 高度扫描（约 187 维）。  
- 为后续蒸馏提供行为克隆目标。

### 1.2 蒸馏阶段目标

- Student 输入严格 52D，与部署一致。  
- 避免 Student 偷偷依赖扫描（扫描不进 policy）。  
- 检查点示例（历史）：`diandog_stairs_omni_student/.../model_8399.pt`。

### 1.3 PPO 微调阶段目标

- 继续盲策略；扫描仅用于：  
  - `feet_clearance` / `feet_foothold` / `feet_miss` / `feet_slide`  
  - Critic 特权观测  
- 修复蒸馏后「不主动抬脚 / 小腿卡沿」等问题。

### 1.4 v1.0 额外目标（本包）

| 问题 | 手段 |
|------|------|
| 无速度原地碎步并漂移 | `stand_still_vel/action/air`，提高 `rel_standing_envs` |
| 平地歪腿、后髋内扣 | `hip_pos`、`rear_hip_pos`、`hip_mirror`、`thigh_pos` |
| 爬楼抬腿不足 | 适度提高 clearance/foothold，加强 miss 惩罚 |
| 课程冲顶过快 | `num_rows=16`，promote_x / min_peak 门控 |

## 2. 任务与配置入口

- Task：`DianDog-Stairs-Omni-Student-RL-v0`  
- EnvCfg：`DianDogStairsOmniStudentFinetuneEnvCfg`（见 `isaac_train/go2_env_cfg.py`）  
- 启动：`scripts/train_diandog_stairs_student_rl.sh`  
- 初始化权重：`STUDENT_CHECKPOINT=.../model_1550.pt`（站立强化前备份）再训至 1200

## 3. 关键奖励项（v1.0）

下列数值来自 `DianDogStairsOmniStudentFinetuneEnvCfg` 快照，实际以 `configs/train_env_snapshot.yaml` 为准。

### 3.1 站立（cmd 范数 < 0.1）

| 名称 | scale（约） | 含义 |
|------|-------------|------|
| stand_still_vel | -3.5 | 惩罚机体系水平速度与 yaw rate |
| stand_still_action | -1.5 | 惩罚动作幅度（促回收默认角） |
| stand_still_air | -1.2 | 惩罚腾空足（抑碎步） |

### 3.2 姿态

| 名称 | scale（约） | 含义 |
|------|-------------|------|
| hip_pos | -3.0 | 全部 haa 偏离默认 |
| rear_hip_pos | -6.0 | 后髋额外惩罚 |
| hip_mirror | -5.0 | fl+fr、rl+rr 和偏离 0 |
| thigh_pos | -1.5 | hfe 偏离默认 |
| knee_pos | -1.0 | kfe 偏离默认 |
| dof_close_to_default | -0.25 | 全关节靠近默认 |

### 3.3 爬楼特权

| 名称 | scale（约） | 含义 |
|------|-------------|------|
| feet_clearance | +0.40 | 摆动相净空跟踪 |
| feet_foothold | +1.1 | 支撑相落足质量 |
| feet_miss | -2.0 | 踩空 |
| feet_slide | -0.55 | 支撑相拖地 |
| feet_air_time | +2.0 | 短腾空奖励（仅 cmd 运动时） |

## 4. 地形与课程

- 子地形：楼梯条带 + 平面，比例约 0.45 / 0.55  
- `step_height_range=(0.02, 0.20)`  
- `num_rows=16` → 等级 0…15 对应高度插值  
- 晋级：`curriculum_stairs_promote_x≈7.0`，相对峰值 `min_peak≈0.12`  
- 楼梯指令：`fixed_command≈(0.85,0,0)`；平面可随机全向

## 5. 仿真与控制时序

```text
Isaac sim.dt = 1/200 = 0.005 s
decimation = 4
→ policy / PD 目标更新周期 = 0.02 s

q_des = q_default + 0.3 * action
set_joint_position_target(q_des)  # DelayedPDActuator Kp=80 Kd=3.5
```

MuJoCo 回放用显式力矩：

```text
τ = 80*(q_des - q) - 3.5*dq
```

## 6. 检查点建议

| 场景 | 推荐 |
|------|------|
| 对外演示 / 默认 | `policy_blind52_iter1200.pt` |
| 对比碎步问题 | 与 `..._pre_standstill.pt` 同场景同指令回放 |
| 继续微调起点 | `iter1200` + 略降 climb 权重，若站立仍抖 |

## 7. 日志关键指标

训练时应关注：

- `Episode_Reward/stand_still_*`：绝对值应变小（惩罚减轻）  
- `Episode_Reward/hip_mirror`：下降  
- `Curriculum/stairs_mean_level`：缓慢上升至 ~15（20 cm）  
- `Mean reward`：难度升高时允许下降  

## 8. 复现命令示例

```bash
# Isaac 续训（示例）
cd <extreme-quadruped-parkour>
GUI=1 NUM_ENVS=4096 \
STUDENT_CHECKPOINT=/path/to/policy_blind52_iter1200.pt \
bash scripts/train_diandog_stairs_student_rl.sh

# MuJoCo
cd mujoco
CKPT=../checkpoints/policy_blind52_iter1200.pt bash play_stairs.sh
```

## 9. 变更记录（摘要）

- 2026-10-30：发布 v1.0；加入站立三项奖励；修复 `stand_still_air` 张量维度 bug；MuJoCo 双楼梯场景与演示视频入库。  
