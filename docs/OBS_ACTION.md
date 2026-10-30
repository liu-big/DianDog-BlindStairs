# 观测与动作对齐手册

保证 Isaac 训练权重在 MuJoCo / 实机部署时 **字节级语义一致**。

## 1. 关节顺序

Isaac `ArticulationData.joint_names`（本机 kit 日志确认）：

```text
0  fl_haa_j1
1  fr_haa_j1
2  rl_haa_j1
3  rr_haa_j1
4  fl_hfe_j2
5  fr_hfe_j2
6  rl_hfe_j2
7  rr_hfe_j2
8  fl_kfe_j3
9  fr_kfe_j3
10 rl_kfe_j3
11 rr_kfe_j3
```

MuJoCo XML 为「每条腿 haa→hfe→kfe」树序，**必须按名称重映射**，见 `mujoco/joint_order.txt`。

## 2. 默认站立角 `_STAND`

与 `isaac_train/diandog.py` 一致：

```python
fl_haa= +0.04, fr_haa= -0.04, rl_haa= -0.04, rr_haa= +0.04
fl_hfe= -0.25, fr_hfe= +0.25, rl_hfe= -0.25, rr_hfe= +0.25
fl_kfe= -0.40, fr_kfe= +0.40, rl_kfe= -0.40, rr_kfe= +0.40
```

观测中的关节位置为 **相对默认角**：`q - q_default`。

## 3. 动作接口

```text
action ∈ R^12   # 网络原始输出（已过 obs normalizer 的输入）
q_des = q_default + action_scale * action
action_scale = 0.3
```

Isaac：`set_joint_position_target(q_des)` + DelayedPDActuator。  
MuJoCo：电机为 torque，脚本侧实现同增益 PD。

## 4. 坐标系约定

| 量 | 约定 |
|----|------|
| 根位置 | 世界系 |
| `root_lin_vel_b` | 机体系线速度 |
| `root_ang_vel_b` | 机体系角速度 |
| `projected_gravity_b` | 重力方向在机体系的单位投影（平地约 `[0,0,-1]`） |
| 指令 | 机体系期望 `(vx, vy, ωz)` |

MuJoCo freejoint：`qvel[0:3]` 为世界系线速度，需用四元数转到机体系；`qvel[3:6]` 为机体系角速度。

## 5. 足端接触

训练：接触力范数 `> 1.0` → 1。  
MuJoCo play：足 body 出现在接触对中 → 1（近似）。  
足顺序：`fl_ft, fr_ft, rl_ft, rr_ft`。

## 6. Obs Normalizer

Checkpoint 内：

```text
obs_normalizer._mean  (1, 52)
obs_normalizer._std   (1, 52)
```

推理：

```text
obs_n = (obs - mean) / (std + 1e-2)
```

与 rsl-rl `EmpiricalNormalization` 一致。

## 7. 加载权重键名

| 来源 | state_dict 键 |
|------|----------------|
| PPO finetune | `actor_state_dict` |
| Distill student | `student_state_dict` |

仅加载 `mlp.*` 与 normalizer；忽略 `distribution.log_std_param`（推理用均值）。

## 8. 常见对齐错误

1. 使用 MuJoCo 树序直接喂网络 → 左右腿交叉错乱。  
2. 忘记减 `default_q` → 观测偏置爆炸。  
3. 线速度未转到机体系 → 指令跟踪全错。  
4. `action_scale` 用成 0.25（Go2 旧默认）→ 步幅偏小。  
5. 归一化 eps 用 `1e-8` 而非 `1e-2` → 与训练分布不一致。  

## 9. 自检脚本思路

1. Isaac dump 一帧 obs（`mujoco/dump_isaac_obs.py`）。  
2. MuJoCo 重置到相同 q/qd/cmd。  
3. 比较 prop 52 维逐维误差；对齐成功时最大绝对误差应很小（接触位可差 0/1）。  
