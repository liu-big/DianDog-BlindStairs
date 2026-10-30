# MuJoCo 对齐与双楼梯场景

## 1. 设计目标

- 与 Isaac DirectRL **同观测、同动作、同 PD 增益、同时序**。  
- 提供可视化友好的 **15 cm + 20 cm** 并排楼梯，便于一次评测两种高度。  
- 网格路径可移植：`meshdir="../assets/meshes"`。

## 2. 关键脚本

| 文件 | 作用 |
|------|------|
| `play_stairs.sh` | 生成场景 + 启动 teleop |
| `gen_stairs_xml.py` | `--dual` 生成 15/20 cm 条带 |
| `play_diandog_stairs_mujoco.py` | 加载 ckpt、构造 52D obs、PD 控制 |
| `diandog_flat.xml` | 机器人与执行器模板 |
| `diandog_stairs.xml` | 生成后的完整场景 |

## 3. 双楼梯几何

```text
Y
▲
│   [15cm strip, blue marker]   y = +lane
│   [ mid flat spawn corridor ] y ≈ 0
│   [20cm strip, orange marker] y = -lane
└──────────────────────────────► X
        start platform → up/down stairs → end
```

参数默认：

- tread = 0.30 m  
- steps/seg = 8  
- start_len = 8.0 m，end_len = 6.0 m  
- chamfer = 0.05 m（缓解直角立面卡脚）  
- foot_r = 0.020 m  
- 可选 `--no-hfe` 关闭大腿盒碰撞（减轻自碰）

## 4. 灯光与地面

- 三盏灯：key / fill / rim，提高录屏亮度。  
- 大地盘 checker，纹理对比提高。  
- 中缝浅色走廊指示出生区。

## 5. 控制循环

```text
for phys_i:
  if phys_i % 4 == 0:
      obs = build_obs()
      action = actor(normalize(obs))
      if ||cmd|| < 0.08: soft-decay action toward 0
      last_action = action
  apply PD torque
  mj_step
  sleep to realtime
```

## 6. 与旧脚本差异（v1.0 重写）

旧版问题：

- `|cmd|` 小时绕过策略并清零动作逻辑不一致；  
- 禁止倒退 / 强制固定前进，导致「退不了、横移怪」；  
- Action EMA 过重，削弱策略输出。  

新版：

- 始终跑 Actor；  
- 完整 SE(2) 遥控；  
- 仅在接近零指令时做轻微衰减，配合训练站立奖励。  

## 7. 故障排查

| 现象 | 排查 |
|------|------|
| 一加载就抽搐 | 检查关节序 / default_q / normalizer |
| 能走不能爬 | 域差；先试 15 cm；检查 chamfer/foot_r |
| 网格丢失 | `meshdir` 是否指向 `assets/meshes` |
| 窗口无响应 | 焦点是否在 MuJoCo；检查 DISPLAY |

## 8. 生成命令

```bash
python gen_stairs_xml.py --dual --no-hfe --foot-r 0.020 \
  --start-len 8 --end-len 6 -o diandog_stairs.xml
```

## 9. 视频

演示文件：`../media/demo_mujoco_blind_climb.mp4`（2026-10-30 录制）。  
