# 观测 / 动作接口规格书

## Actor 输入

- 类型：`float32`  
- 形状：`(N, 52)` 或部署 `(52,)`  
- 归一化：`(x - mean) / (std + 1e-2)`，mean/std 存于 ckpt  

## Actor 输出

- 类型：`float32`  
- 形状：`(N, 12)`  
- 语义：相对默认角的残差，再乘 `action_scale=0.3`  

## 默认关节角 q0（Isaac 顺序）

```
haa:  +0.04, -0.04, -0.04, +0.04   # fl fr rl rr
hfe:  -0.25, +0.25, -0.25, +0.25
kfe:  -0.40, +0.40, -0.40, +0.40
```

## 指令范围（Play 建议）

| 分量 | 建议范围 | 训练楼梯常用 |
|------|----------|--------------|
| vx | [-1.2, 1.2] | 固定约 0.85 前进 |
| vy | [-0.7, 0.7] | 平地 omni |
| yaw | [-1.2, 1.2] | 平地 omni |

零指令阈值：`||cmd||_∞` 意义下 xy 与 yaw 均 `< 0.1` 视为站立门控。

## 足端接触

训练：接触传感器历史最后一帧力模 `> 1.0` → 1。  
MuJoCo play：足 body 出现在接触列表 → 1（近似）。

## 检查点键

```
actor_state_dict:
  mlp.0.weight ... mlp.6.bias
  obs_normalizer._mean / _std / _var / count
  distribution.log_std_param   # 推理可忽略
```

推理仅需 `mlp.*` + `obs_normalizer._mean/_std`。
