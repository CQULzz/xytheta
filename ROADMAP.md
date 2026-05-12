# Roadmap

## Phase 0: 项目初始化与工程化整理

- [x] 整理 README 框架
- [x] 补充项目管理规范
- [x] 补充 CHANGELOG、TODO、ROADMAP
- [x] 补充实验记录模板
- [x] 检查并补充 `.gitignore`

可交付结果：仓库结构清晰，后续开发者可以理解如何运行、开发、记录实验和发布版本。

当前优先级：最高。

## Phase 1: 基础环境稳定

- [x] 保留 `Xytheta-Transa-v0`
- [x] 新增 `Xytheta-Transa-v1`
- [ ] 梳理环境参数配置方式
- [ ] 补充最小 smoke test 文档
- [ ] 确认多环境 `num_envs` 下 CSV 记录稳定

可交付结果：v0/v1 都能稳定启动，环境配置和运行方式明确。

当前优先级：高。

## Phase 2: 实验与奖励优化

- [ ] 记录 v1 reward-guided rollout 基线
- [ ] 对比 random、demo、reward-guided、PPO policy
- [ ] 调整探索奖励网格分辨率、LiDAR 参数和靠墙惩罚权重
- [ ] 形成实验日志和结果图表

可交付结果：可复现实验记录、对比图、关键参数说明。

当前优先级：中。

## Phase 3: 训练与评估流程

- [ ] 整理 RL-Games PPO 训练命令
- [ ] 规范 checkpoint 保存和忽略规则
- [ ] 增加评估脚本或评估说明
- [ ] 形成稳定训练配置

可交付结果：从训练到评估的最小闭环。

当前优先级：中。

## Phase 4: 稳定版本发布

- [ ] 清理临时代码和调试输出
- [ ] 更新 README、CHANGELOG 和 VerDairy
- [ ] 确认 main 分支可运行
- [ ] 打 tag 标记稳定版本

可交付结果：可复现、可交付、可回滚的稳定 release。

当前优先级：低。
