# Skill-Learning VLA 参考文献登记表

> 主智能体维护的汇总表。仅收录已核对题名与来源的工作；详细笔记由各主题文件提供。

## 收录准则

- 优先论文原文、作者项目页、正式会议页面和官方代码仓库。
- 每项明确其“skill”语义，避免把语言规划、动作分块和潜技能混为一谈。
- 重点记录相对直接行为克隆/VLA 的增益是否来自低样本设置，以及是否验证跨任务复用。
- 预印本与未复现的强结论明确标注，不把发布日期误写为会议录用年份。

## 主题索引

| 主题 | 子智能体文件 | 状态 |
|---|---|---|
| 潜技能发现与层级模仿学习 | `references_latent_skills.md` | 第一轮完成 |
| 层级 VLA、语言规划与长时程控制 | `literature_hierarchical_vla.md` | 第一轮完成 |
| 动作 tokenizer、chunk 与跨本体表示 | `references_action_representation.md` | 第一轮完成 |

## 对课题的影响标签

- `核心继承`：可直接成为方法组件或主要基线。
- `关键对照`：必须实验比较，否则创新性或结论不足。
- `评测借鉴`：用于任务拆分、指标或样本效率协议。
- `风险提示`：揭示负迁移、表示坍缩、精度损失或工程不可行性。
- `外围背景`：用于研究版图，不承担核心论证。

## 核心论文表

| 工作 | 年份 / venue | Skill 语义 | 对课题影响 | 标签 |
|---|---|---|---|---|
| LISA | 2022 / NeurIPS | 语言对齐离散 skill | 语言可解释性与低样本组合先例；固定边界、精细控制不足 | 关键对照 |
| VQ-BeT | 2024 / ICML | 离散 latent action chunk | 量化动作基线；证明压缩不等于语义技能 | 关键对照 |
| QueST | 2024 | 结构化可迁移 skill | 连续/离散技能迁移的重要近邻 | 核心继承 |
| PRISE | 2024 | BPE 可变长动作技能 | 排除“可变长度 token”作为单独创新 | 关键对照 |
| STAR | 2025 / ICML Spotlight | residual VQ skill + refinement | 最接近竞争工作；连续精修对精细任务不可缺 | 核心继承 |
| SkillDiffuser | 2024 / CVPR | 语言对齐 skill + diffusion | 高层生成式技能规划对照 | 关键对照 |
| LAPA | 2025 / ICLR | 视频 latent action | 可用于无动作视频预训练，但不等于闭环 skill | 外围背景 |
| Moto | 2025 / ICCV | latent motion token | 视频到机器人迁移先例；需超越视觉变化 token | 外围背景 |
| CompILE | 2019 / ICML | 可学习事件边界 | 事件驱动可变边界的经典概念基线 | 核心继承 |
| OPAL | 2021 / ICLR | 连续 primitive latent | 离散技能价值的关键反事实 | 关键对照 |
| SkillVLA | 2026 / 预印本 | 左右臂技能复用与协作 gate | Mobile ALOHA 双臂主张的直接竞争；需超越按臂解耦 | 关键对照 |
| GPLA | 2026 | 层级语言子任务与动作显式对齐 | 若技能带语言解释，需检验解释忠实性而非只看词义 | 风险提示 |
| SuSIE | 2024 / ICLR | 可达中间视觉子目标 | 层级提高精细度的直接证据；应设 oracle subgoal 对照 | 核心继承 |
| RT-H | 2024 | 语言动作层级 | 总体提升但精细任务动作簇更好，证明语言信息瓶颈 | 关键对照 |
| Hi Robot | 2025 / ICML | 语言原子技能 + 低层 VLA | 长时推理先例，但依赖分段标签且非 few-shot 证据 | 关键对照 |
| FAST | 2025 | DCT+BPE 动作 token | 高效压缩基线；可变长 token 不是技能创新 | 关键对照 |
| ACT | 2023 / RSS | 固定动作 chunk | ALOHA 精细双臂首要基线；检验事件边界 | 关键对照 |
| Diffusion Policy | 2023 / RSS | 连续多峰动作 chunk | 检验离散技能是否损害精度的强连续基线 | 关键对照 |
| CrossFormer | 2024 | 共享骨干 + 本体专用动作头 | 支持高层共享、低层本体专用；需报告负迁移 | 核心继承 |
| UniVLA | 2025 / RSS | task-centric 离散 latent action | 最大创新冲突；固定约 1 秒 token 的强对照 | 关键对照 |
| CLAM | 2025 | 连续 latent action | 离散化精度风险的最强反证 | 风险提示 |
| UHAS / Cloak | 2026 / 预印本 | 几何 canonicalization / 视觉遮蔽 | 跨本体收益的替代解释，不一定来自 skill | 风险提示 |
| UniAct | 2025 / CVPR | universal atomic action + 本体 decoder | 与共享离散 skill/本体解码高度重合 | 关键对照 |
| LAD / OPFA | 2025-2026 | 几何对齐连续跨本体 latent | 跨本体少样本主张已拥挤，需拆分几何与语义收益 | 风险提示 |
| Transferring Contact, Not Just Motion | 2026 / 预印本 | 标定 force-position 跨手接口 | 已占据跨本体 contact 表示；不能声称首次 | 关键对照 |
