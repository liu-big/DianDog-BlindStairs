# DianDog-BlindStairs-v1.0

**点犬（DianDog）· 52D 盲爬楼梯策略完整发布包**

> 版本代号：`Blind Climb Standstill`  
> 发布日期：2026-10-30  
> 主策略文件：[`checkpoints/policy_blind52_iter1200.pt`](checkpoints/policy_blind52_iter1200.pt)  
> 演示视频：[`media/demo.mp4`](media/demo.mp4)

---

## 演示视频

MuJoCo 双楼梯场景（15cm / 20cm）盲策略遥控评测（约 38s，H.264）。

### 在线观看 / 下载（推荐 GitHub）

- **GitHub 在线打开（可播放）**：[media/demo.mp4](https://github.com/liu-big/DianDog-BlindStairs/blob/main/media/demo.mp4)
- **GitHub 直链下载**：https://raw.githubusercontent.com/liu-big/DianDog-BlindStairs/main/media/demo.mp4
- **Gitee 文件页**：[media/demo.mp4](https://gitee.com/yonheybot/diandog-blindstairs-v1.0/blob/master/media/demo.mp4)  
  （Gitee raw 常 403，请用文件页点「下载」或 Clone）

### 仓库内相对路径

[`media/demo.mp4`](media/demo.mp4) · 同内容备份名 [`media/demo_mujoco_blind_climb.mp4`](media/demo_mujoco_blind_climb.mp4)

```bash
# Clone 后本地播放
xdg-open media/demo.mp4
```

---

## 一句话介绍

点犬 **无高度扫描（52D 本体感知）** 盲爬楼梯策略：Isaac Lab DirectRL 蒸馏 + PPO 微调，课程最高台阶 **20cm**；强化零指令站立（抑制碎步/漂移）与髋姿态对称；附 **MuJoCo** 对齐回放、网格资源、USD、训练环境源码与配置快照。

---

## 目录结构

```
DianDog-BlindStairs-v1.0/
├── README.md                          # 本说明
├── media/
│   └── demo_mujoco_blind_climb.mp4    # 演示视频
├── checkpoints/
│   ├── policy_blind52_iter1200.pt     # ★ 主推策略（站立强化后）
│   └── policy_blind52_iter1550_pre_standstill.pt  # 对照（站立强化前）
├── params/                            # 训练 run 导出的 env/agent 快照
├── configs/                           # 可读 YAML 配置快照 + 关节命名
├── assets/
│   ├── meshes/                        # STL 碰撞/可视化网格（MuJoCo）
│   └── usd/                           # Isaac 用 USD 资源
├── mujoco/                            # MuJoCo 双楼梯仿真与遥控
├── isaac_train/                       # DirectRL 环境与机器人配置源码
├── scripts/                           # Isaac 训练 / Play 启动脚本
└── docs/                              # 电阻/装配备注、架构说明等
```

---

## 策略规格

| 项目 | 规格 |
|------|------|
| 机器人 | DianDog 点足四足 |
| 策略类型 | 盲策略（无 height scan 入 policy） |
| 观测维度 | **52D**：关节相对位置 12 + 关节速度 12 + 投影重力 3 + 机体系线速度 3 + 机体系角速度 3 + 速度指令 3 + 上一步动作 12 + 足端接触 4 |
| 动作维度 | **12D** 关节位置残差 |
| `action_scale` | `0.3` |
| 默认站立角 | haa ±0.04 / hfe ±0.25 / kfe ±0.40（与 `diandog.py` `_STAND` 一致） |
| PD | `Kp=80`, `Kd=3.5`（DelayedPDActuator） |
| 仿真步长 | `dt=0.005`，`decimation=4`（控制周期 20ms） |
| 楼梯课程上限 | **20 cm** 台阶高度 |
| Critic / Reward | 训练期可使用特权高度扫描；部署仅 52D |

关节顺序（Isaac PhysX `joint_names`，**非** MuJoCo XML 树序）：

```
fl_haa_j1, fr_haa_j1, rl_haa_j1, rr_haa_j1,
fl_hfe_j2, fr_hfe_j2, rl_hfe_j2, rr_hfe_j2,
fl_kfe_j3, fr_kfe_j3, rl_kfe_j3, rr_kfe_j3
```

详见 [`mujoco/joint_order.txt`](mujoco/joint_order.txt)。

---

## 训练脉络

1. **Teacher（239D）**  
   策略含高度扫描，打通楼梯课程。

2. **DAgger 蒸馏 → 52D Student**  
   学生仅本体感知；扫描保留给教师 / 特权通道。

3. **PPO 微调（Student-RL）**  
   - 特权奖励：抬腿净空、落足、防踩空、防拖地  
   - 姿态：`hip_pos` / `rear_hip_pos` / `hip_mirror` / `thigh_pos` / `knee_pos`  
   - 课程：`step_height_range=(0.02, 0.20)`，多行缓升

4. **v1.0 Standstill 强化（本包主权重）**  
   从约 `model_1550` 续训，增加零指令站立项：  
   - `stand_still_vel`：惩罚站立时基座水平/偏航速度（防漂移）  
   - `stand_still_action`：惩罚站立时动作幅度（防抽搐）  
   - `stand_still_air`：惩罚站立时抬脚（防碎步）  
   并提高 `rel_standing_envs`，最终得到 **`policy_blind52_iter1200.pt`**。

---

## 快速开始：MuJoCo 评测

### 依赖

- Python 3.10+（建议与 Isaac 同环境 `isaaclab51`）
- `mujoco`, `torch`, `numpy`

### 运行

```bash
cd mujoco
CKPT=../checkpoints/policy_blind52_iter1200.pt bash play_stairs.sh
```

脚本会生成 **15cm（+Y，蓝标）** 与 **20cm（-Y，橙标）** 并排楼梯，出生在中间平道。

### 键位（焦点在 MuJoCo 窗口）

| 键 | 功能 |
|----|------|
| `8` / `2` | 前进 / 后退 |
| `7` / `9` | 左移 / 右移（进 15cm / 20cm 车道） |
| `4` / `6` | 左转 / 右转 |
| `0` / `5` | 停止（cmd=0） |
| `R` | 重置 |
| `P` | 暂停 |

### 对齐说明

MuJoCo 脚本按 Isaac DirectRL 对齐：

- 观测构造与 `go2_env.py` `_get_observations` 一致  
- `q_des = default_q + action_scale * action`  
- `τ = Kp*(q_des-q) - Kd*dq`  
- 零指令时 play 侧额外衰减动作，配合训练站立奖励  

> **域差提示**：Isaac PhysX 与 MuJoCo 接触不同，高台阶迁移可能弱于 Isaac 内表现；平地走停与遥控请以本包脚本评测。

---

## Isaac Lab 评测 / 续训

任务注册名：

| 用途 | Task ID |
|------|---------|
| 盲策略 Play | `DianDog-Stairs-Omni-Student-RL-Play-v0` |
| 盲策略训练 | `DianDog-Stairs-Omni-Student-RL-v0` |

启动脚本（需完整 Isaac Lab / parkour 工作区）：

```bash
# 训练示例
STUDENT_CHECKPOINT=/path/to/policy_blind52_iter1200.pt \
  bash scripts/train_diandog_stairs_student_rl.sh

# Isaac GUI 遥控
CKPT=/path/to/policy_blind52_iter1200.pt \
  bash scripts/play_diandog_stairs_omni.sh
```

本包 `isaac_train/` 为环境源码快照（`go2_env.py` / `go2_env_cfg.py` / `diandog.py`），便于对照奖励与课程配置；合入原仓库时注意路径与 `PYTHONPATH`。

机器人 USD 见 [`assets/usd/`](assets/usd/)，配置入口 [`isaac_train/diandog.py`](isaac_train/diandog.py)。

---

## 资源文件说明

### `assets/meshes/`

点犬各连杆 STL（`base_link`、四腿 `haa/hfe/kfe/ft`），供 MuJoCo `meshdir=../assets/meshes` 加载。

### `assets/usd/`

Isaac 侧资产导出，用于 DirectRL 中 `DIANDOG_CFG` 的 `usd_path`（合入工程后请按本机路径修改）。

### `configs/`

- `train_env_snapshot.yaml` / `train_agent_snapshot.yaml`：本次成功 run 的配置快照  
- `joint_names_diandog.SLDASM.yaml`：SolidWorks 装配关节命名

### `docs/dianzu.txt`

电机/电阻等硬件侧备注，便于软硬件对照。

---

## 检查点选用建议

| 文件 | 适用 |
|------|------|
| **`policy_blind52_iter1200.pt`** | 默认部署；停步更稳、姿态约束更强 |
| `policy_blind52_iter1550_pre_standstill.pt` | 对照实验；站立强化前，碎步更明显 |

加载时优先读取 `actor_state_dict`（PPO）；若为蒸馏权重则读 `student_state_dict`。MuJoCo 脚本已自动兼容两者。

---

## 复现清单（Checklist）

- [ ] Clone 本仓库并确认 `media/demo_mujoco_blind_climb.mp4` 可播放  
- [ ] `cd mujoco && CKPT=../checkpoints/policy_blind52_iter1200.pt bash play_stairs.sh`  
- [ ] 零指令观察是否明显减少碎步/漂移  
- [ ] `7` 进 15cm、`9` 进 20cm，`8` 前进评测爬楼  
- [ ] （可选）Isaac Play 同权重对比 PhysX 表现  

---

## 已知限制

1. MuJoCo 直角立面接触与 PhysX 不一致，首级台阶易卡脚；场景已加倒角与脚半径微调。  
2. 楼梯课程在 Isaac 内顶满 20cm 不等于 MuJoCo 同等成功率。  
3. 零指令站立依赖训练分布中的 `rel_standing_envs`；极端扰动下仍可能小幅晃动。  
4. 横移/倒退在楼梯任务中采样较少，平地 omni 能力以指令跟踪为准。  

---

## 版本与署名

- **版本名**：`DianDog-BlindStairs-v1.0`  
- **Gitee**：`git@gitee.com:yonheybot/diandog-blindstairs-v1.0.git`  
- **作者标识**：19892940973 / yaoguang816@qq.com  
- 训练栈基于 Isaac Lab DirectRL 与 extreme-quadruped-parkour 适配；点犬资产与本发布包按项目约定使用。

---

## 许可证说明

本仓库用于点犬盲爬楼梯策略发布与仿真评测。第三方依赖（Isaac Lab、MuJoCo、PyTorch 等）遵循其各自许可证。上传与二次分发时请保留本 README 与资源署名信息。
