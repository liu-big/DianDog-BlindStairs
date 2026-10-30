# Changelog

## DianDog-BlindStairs-v1.0 — 2026-10-30

### Added
- 主策略 `policy_blind52_iter1200.pt`（站立约束 + 爬楼微调）
- 对照策略 `policy_blind52_iter1550_pre_standstill.pt`
- MuJoCo 双楼梯场景（15 cm / 20 cm）与 Isaac 对齐回放脚本
- 演示视频 `media/demo_mujoco_blind_climb.mp4`
- 完整 meshes / usd 资源
- DirectRL 环境源码快照与训练脚本快照
- 文档：TRAINING / OBS_ACTION / MUJOCO_ALIGN / REWARD_AND_ENV_REFERENCE

### Fixed
- `stand_still_air` 使用错误 contact 张量维度导致训练崩溃
- MuJoCo 旧脚本禁止倒退/横移、过度 EMA 导致遥控异常

### Changed
- 零指令站立奖励三项；姿态镜像髋加强；课程最高 20 cm
