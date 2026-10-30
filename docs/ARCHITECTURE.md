# DianDog BlindStairs 架构说明

## 1. 系统分层

```
┌─────────────────────────────────────────────────────────────┐
│  Teleop / Play（Isaac GUI 或 MuJoCo Viewer）                 │
│  指令 cmd = (vx, vy, yaw_rate)                               │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Policy MLP（52 → 512 → 256 → 128 → 12）                     │
│  + EmpiricalNormalization（ckpt 内 mean/std）                │
└───────────────────────────┬─────────────────────────────────┘
                            │ action a ∈ R^12
┌───────────────────────────▼─────────────────────────────────┐
│  目标角 q* = q0 + 0.3 * a                                   │
│  力矩 τ = 80*(q*-q) - 3.5*dq                                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  物理引擎：Isaac PhysX（训练） / MuJoCo（迁移评测）            │
└─────────────────────────────────────────────────────────────┘
```

训练期 Critic 可额外接收特权高度扫描与质量/摩擦等 DR 特征；Actor 部署保持 52D。

## 2. 观测布局（52D）

| 切片 | 维数 | 含义 |
|------|------|------|
| 0:12 | 12 | `joint_pos - default_joint_pos` |
| 12:24 | 12 | `joint_vel` |
| 24:27 | 3 | `projected_gravity_b` |
| 27:30 | 3 | `root_lin_vel_b` |
| 30:33 | 3 | `root_ang_vel_b` |
| 33:36 | 3 | `commands` |
| 36:48 | 12 | `last_action` |
| 48:52 | 4 | foot contact flags（力模 > 1） |

## 3. 奖励模块（Student Finetune）

### 运动跟踪

- `track_lin_vel_xy_exp` / `track_ang_vel_z_exp`
- `lin_vel_z_l2` / `ang_vel_xy_l2`

### 爬楼特权（仅训练）

- `feet_clearance`：摆动腿相对地面净空  
- `feet_foothold`：支撑腿垂直力占比  
- `feet_miss`：踩空惩罚  
- `feet_slide`：支撑相拖地  

### 姿态

- `hip_pos` / `rear_hip_pos` / `hip_mirror` / `thigh_pos` / `knee_pos`  
- `flat_orientation_l2` / `base_height` / `upward`

### 站立（v1.0）

当 `||cmd_xy|| < thr` 且 `|cmd_yaw| < thr`：

- `stand_still_vel`  
- `stand_still_action`  
- `stand_still_air`  

## 4. 课程

Terrain generator：`MeshStairsStrip` + `MeshPlane`，`curriculum=True`。  
难度行插值 `step_height` 从 2cm → 20cm。  
晋升条件：前进距离过峰 + 相对爬升峰值门槛；平坦列不参与楼梯均值膨胀。

## 5. 仿真对齐要点

1. 关节名映射必须用 Isaac 顺序，不能按 MuJoCo body 树顺序直接喂网络。  
2. MuJoCo freejoint：线速度世界系需转到机体系；角速度取 body 系。  
3. 四元数均为 wxyz。  
4. 足端接触 play 用二值接触近似训练力阈值。  

## 6. 文件映射

| 仓库路径 | 原工程角色 |
|----------|------------|
| `isaac_train/go2_env.py` | DirectRL 环境步进/奖励/课程 |
| `isaac_train/go2_env_cfg.py` | 任务与 StudentFinetune 配置 |
| `isaac_train/diandog.py` | 机器人 ArticulationCfg |
| `mujoco/play_diandog_stairs_mujoco.py` | 对齐回放 |
| `mujoco/gen_stairs_xml.py` | 双楼梯场景生成 |
| `checkpoints/*.pt` | rsl-rl actor 权重 |
