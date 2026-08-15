# 面向精细操作的高样本效率 Skill-Learning VLA 研究方案

> 工作报告，随 `/goal` 调研持续更新。  
> 版本：v0.2，2026-08-14  
> 暂定方法名：CASK（Contact-Aware Shared Skills）
> 详细模型规格与设计理由：`model_design.md`

## 摘要

本课题研究一个可证伪的问题：**在多任务精细操作演示中，能否通过事件驱动分段和跨任务“对象后果 + 接触模式”一致性，学习可变时长的共享潜技能，并用状态条件连续精修，使新任务以更少示教获得更高成功率？**

建议不从头训练新的通用大模型，而以 `openpi` 的 π0/π0.5 flow policy 作为 VLA 基座，构成“语言子任务 -> 视觉/几何子目标 -> 接触感知动作技能”的三层结构。技能表示采用“离散 mode + 连续 residual + termination”，离散部分承担跨任务复用与组合，连续部分保留窄容差操作所需的位姿和接触细节。论文的主要证据不是 token 可视化，而是 held-out 新任务上的 demos-to-success 曲线、技能反事实交换、负迁移检查与精细误差指标。

## 1. 问题定义

### 1.1 输入与目标

给定多任务演示集合

\[
\mathcal D=\{(o_{1:T},s_{1:T},a_{1:T},l)\},
\]

其中 `o` 为多视角 RGB（可选深度/触觉），`s` 为本体与对象关系状态，`a` 为连续机器人动作，`l` 为任务语言。学习：

- 边界模型 `b_t`：决定技能何时开始/终止；
- 技能编码器 `q(z_k | segment_k)`：输出跨任务共享的离散 `z_k`；
- 连续实例码 `r_k`：保存对象几何、目标位姿与接触强度等细节；
- 高层策略 `π_H(z_k, r_k, termination | o,s,l,history)`；
- 低层闭环策略 `π_L(a_{t:t+h} | o,s,z_k,r_k)`。

目标是使少样本新任务仅需学习技能选择/组合与小型 adapter，而尽量复用低层技能。

### 1.2 “技能”的操作性定义

一个技能不是固定 `K` 帧动作的别名。本文建议定义技能为：

> 在对象中心状态和接触上下文下，具有可预测终止条件、并导致稳定语义/几何后果的闭环时序行为模式。

因此，技能复用至少需要满足：

- 同一技能在多个任务中出现，而非与 task ID 一一对应；
- 交换到兼容上下文后仍可执行，而非只在 latent probe 上聚类良好；
- 对段末对象状态变化或接触模式有预测力；
- 其低层控制误差不因离散量化而破坏精细任务。

## 2. 与现有研究的关系

### 2.1 四类容易混淆的抽象

| 抽象 | 典型作用 | 与本课题关系 |
|---|---|---|
| 语言/视觉 task plan | 任务分解与长时程规划 | 可作为高层先验，但不能替代精细控制 |
| semantic skill | 跨任务复用、可组合子策略 | 本课题的核心对象 |
| latent action / tokenizer | 压缩动作或视觉变化 | 可作实现手段，但压缩码未必是技能 |
| motor primitive | 高频闭环、接触与顺应控制 | 本课题低层执行所需 |

### 2.2 最接近的工作与剩余空间

- LISA、SkillDiffuser：证明语言对齐的离散技能有助于组合，但真实精细控制证据有限，边界多为固定窗口。
- VQ-BeT、QueST、PRISE：分别覆盖 latent action、结构化可迁移表示和 BPE 可变时长技能；因此“VQ/BPE 动作 token”不能单独构成创新。
- STAR：已覆盖残差量化、因果技能组合和连续 action refinement；其消融显示去除 refinement 会使 LIBERO-Long 从 88.5% 降至 37.6%，直接说明纯离散技能不适合承担精细执行。
- LAPA、Moto：用无动作视频 latent 连接视频和机器人，但 visual transition token 不等于带终止条件的可执行技能。
- SkillMimicGen：展示显式技能分段、重定向和拼接对示教扩增有效，提示本课题可把技能抽象与数据生成结合，但它依赖已有分段/对象信息。
- HiRT、RT-H、HAMSTER 等层级策略：主要改善多层时间尺度或 waypoint/action hierarchy；它们是层级结构基线，不能直接证明无监督共享技能。
- SkillVLA（2026 预印本）：直接研究真实双臂的左右臂 skill reuse；报告通用 π0.5/π0-FAST 在未见左右组合上为 0%，其显式按臂推理与协作 gate 达到 51%，并在 continual setting 中用 5 条新双臂示教接近峰值。若本课题以 Mobile ALOHA 双臂重组为核心，它是最直接竞争工作。CASK 必须通过自动事件分段、对象后果/接触一致 latent 和窄容差精度超出“按臂解耦 + 协作模式切换”。
- GPLA（2026）：指出层级 VLA 自生成语言子任务可能与实际视觉和动作不一致，并用 action-conditioned contrastive grounding 与 preference learning 做显式对齐。它提示 CASK 若给 skill token 附语言解释，必须单独评测“语言是否忠实描述动作/后果”；解释性不能仅靠观察 code 对应的高频词。
- UniVLA（RSS 2025）：已覆盖 task-centric 离散 latent action、跨视频/机器人预训练和少量目标动作 decoder；论文报告 LIBERO-Goal 用 10% 示范达到 86.3%，高于 OpenVLA 全数据的 79.2%。这是 CASK 最大创新冲突，必须用事件边界、contact/outcome 一致性和连续 residual 明确区分。
- CLAM（2025）：以连续 latent action 明确反对先离散化精细动作，并在模拟/真机报告相对 latent-action baselines 的明显收益。它要求本课题把 `continuous only` 设为核心反事实；若 `z+r` 不优于连续 latent，离散共享 mode 的必要性不成立。
- CrossFormer、UHAS 与 Cloak 提醒：跨本体改善可能来自共享视觉骨干、本体专用头、几何 canonicalization、IK retargeting 或遮蔽 embodiment shortcut，并不自动证明共享 skill。主论文不宜把跨本体作为核心结论。
- CVPR 2025 UniAct 已直接提出 universal discrete atomic behavior codebook 与本体专用 decoder；LAD/OPFA 已研究连续或几何感知跨本体 latent。因此“共享离散动作 + 新机器人轻量 decoder”也不能作为本论文核心创新。
- 《Transferring Contact, Not Just Motion》（2026 预印本）已用标定的 joint torque/fingertip force 与共享 pose latent 做跨灵巧手迁移，平均 reach-and-lift 从 0.44 提至 0.71。CASK 不能声称首次引入 contact consistency；区别只能是从多任务示范中用接触/后果事件学习技能边界和离散语义，而非统一物理力坐标。

当前较可辩护的缺口是：**很少有工作同时要求 latent skill 对跨任务的对象后果/接触模式保持一致、具有事件驱动可变时长，并通过状态条件连续精修在窄容差 few-shot 操作上验证。**

## 3. 建议方法：CASK

### 3.1 总体架构

```text
RGB / proprio / optional force + language
                  |
       semantic planner: short language subtask
                  |
       reachable visual/geometric subgoal
                  |
       object-centric trajectory encoder
                  |
       event boundary predictor ------ termination
                  |
       segmented trajectory encoder
                  |
       discrete shared skill z + continuous instance residual r
                  |
        high-level skill policy (VLA)
                  |
       π0 flow action expert conditioned on z, r
                  |
       short closed-loop action chunk
```

三层的职责必须分开：语言层表达“做什么”并支持人类干预；视觉/几何子目标表达“到哪里、形成何种对象关系”，缩短低层控制距离；latent skill 表达“如何稳定做到”。SuSIE 的 oracle 对照显示，中间视觉子目标可比直接给最终真值目标更有效；RT-H 则显示语言层在最精细接触任务上可能不如动作聚类，二者共同说明不能用语言短语替代低层技能 latent。

### 3.2 事件驱动边界

边界候选由以下信号联合产生：

- 夹爪开闭或双臂角色切换；
- 末端速度驻点、方向突变、动作 jerk；
- 对象相对末端位姿变化的 regime shift；
- 力/力矩/触觉接触变化；若无传感器，以视觉遮挡、对象开始随动和动作停顿作为代理。

边界头输出校准概率，并加入边界稀疏与合理持续时间先验；默认实现用最短/最长时长、NMS 和 merge 形成可审计 hard segments，不声称 action loss 可穿过这些硬操作。若需要联合改进，先采用交替重分段；soft/Gumbel boundary 仅作为独立实验。若数据没有人工边界，可对少量轨迹标注接触事件，仅用于验证 boundary F1，不必完全监督训练。

### 3.3 混合技能表示

每段编码为 `z_k` 与 `r_k`：

- `z_k`：FSQ、residual VQ 或小型 codebook，表示 approach、grasp、align、insert、handover 等可共享 mode；
- `r_k`：连续 residual，表示具体对象尺寸、目标相对位姿、速度/力度和轨迹微调；
- decoder 显式输入当前 proprioception、对象相对位姿和 embodiment mask，使同一技能在不同构型下可实例化。

该结构回应 STAR 的精度风险，也避免让离散码背负全部控制信息。

### 3.4 跨任务共享约束

总损失可写为：

\[
\mathcal L = \mathcal L_{flow/action}
+ \lambda_p \mathcal L_{prior}
+ \lambda_q \mathcal L_{quant}
+ \lambda_o \mathcal L_{outcome}
+ \lambda_c \mathcal L_{contact}
+ \lambda_x \mathcal L_{cross-task}
+ \lambda_b \mathcal L_{boundary}
+ \lambda_t \mathcal L_{termination}
+ \lambda_g \mathcal L_{subgoal}
+ \lambda_i \mathcal L_{invariance}.
\]

- `L_prior`：将可看完整轨迹的 `q(z,r|segment)` 蒸馏到只看当前/历史上下文的在线 `p(z,r|context)`；主评测只能使用后者；
- `L_outcome`：预测 segment 末端的对象中心状态 delta；
- `L_subgoal`：预测短时可达的关键点/相对位姿或未来视觉 latent，并用 inverse dynamics/reachability 约束可执行性；
- `L_contact`：预测接触类别、夹爪事件或力变化；
- `L_cross-task`：跨任务中后果相同的 segment 拉近，动作相似但后果不同的 segment 推远；
- `L_termination`：训练严格因果的在线 hazard；离线双向边界模型只用于训练分段，不能在部署时读取未来；
- `L_invariance`：用 task-adversarial head 或条件互信息约束减少 code 泄漏 task ID；
- `L_flow/action`：保留 π0 连续动作 flow matching，不用离散码直接替代精细动作头。

正负对构造必须避免伪标签循环。优先依据可测的对象状态变化、接触事件和少量人工审计，而非仅按动作距离聚类。

### 3.5 接入当前 `openpi`

最低风险的实现顺序：

1. 保持 PaliGemma/SigLIP 视觉语言前缀和 π0 action expert；先冻结大部分 backbone。
2. 离线训练 boundary + skill tokenizer，输入数据管线输出的 canonical state/action 与有效维 mask。
3. 在 `Pi0.embed_suffix` 前加入 skill condition token 或通过 AdaRMS/FiLM 调制 action expert。
4. 增加 skill/termination 辅助头，并使低层 flow loss继续在本体原生有效动作维上计算。
5. 新任务 few-shot 阶段优先只训练高层 skill predictor、residual adapter 和 LoRA，再与全量微调比较。

重要工程决策：128 维 canonical 表示优先用于 **skill encoder 的跨本体对齐**；低层执行维持本体原生有效动作语义。当前 openpi checkpoint 的 32 维动作接口通过 slot mask 装入/解包原生动作，flow loss 只在有效槽位计算，避免让 padding 和不同控制语义污染 π0 动作头。canonical feature mask 在投影与 loss 中逐维生效，时间 padding mask 才进入 attention key-padding；二者不能混用。

### 3.6 详细规格补充

`model_design.md` 将本节细化为可实现规格，补充了对象中心输入、离线边界与在线 termination 的区分、轨迹后验与在线先验的蒸馏、FSQ 离散 mode、16 维连续 residual、grounded subgoal、π0 条件注入、梯度隔离、训练阶段、因果 skill swap 和停止门槛。该文档是后续实现的模型规格来源；本报告继续承担研究问题、创新边界和整体实验论证。

## 4. 实验设计

### 4.1 任务拆分

建议使用“基础技能预训练任务”和“组合式 held-out 精细任务”拆分，而不是随机 episode 拆分。

- 预训练任务覆盖 approach、grasp、transport、align、insert/press/release 的多种组合和对象。
- 新任务保留部分已见技能但使用未见组合、对象或目标几何。
- 至少设置一组真正新技能任务，测量何时共享会负迁移。

模拟首选 LIBERO + robosuite/MimicGen 的 nut assembly、square/three-piece assembly、tool hanging 或自定义 peg insertion。真实 Mobile ALOHA 选 2-3 个共享阶段明显的任务，例如插接、扣合、精确放置/装配；每个新任务采集 1/5/10/20/50 条演示。

### 4.2 必要基线

| 类型 | 基线 |
|---|---|
| 直接策略 | ACT、Diffusion Policy、π0/π0.5 LoRA/full FT |
| 连续 latent | OPAL/Play-LMP 风格 |
| 离散固定窗 | VQ-BeT、QueST/STAR 风格 |
| 可变长度 | PRISE/BPE、CompILE 风格 |
| 语言/层级 | LISA 或 SkillDiffuser、HiRT/RT-H（按实现可用性选） |
| 双臂复用 | SkillVLA；独立双单臂 expert；单一拼接动作 π0.5/π0-FAST |
| 动作表示 | FAST、ACT、Diffusion Policy、UniVLA、CLAM |
| 跨本体扩展 | CrossFormer、本体专用低层头、UHAS/Cloak 式几何/视觉对齐 |
| 本方法消融 | 无边界、无 outcome/contact、无 residual、无 task-invariance、随机 code |

公平性要求：相同预训练数据、视觉 backbone、action horizon、参数/训练步预算和新任务示教数。否则不能把大模型或更多数据带来的收益归因于技能。

### 4.3 核心指标

- 主指标：任务成功率及 95% bootstrap 置信区间；每条件建议至少 50 次 rollout，真实机器人资源不足时至少 20 次并报告区间。
- 样本效率：1/5/10/20/50 demos 的成功率曲线、AUC、达到 70% 或预注册阈值所需示教数。
- 精细度：末端位置/角度误差、插入深度、接触峰值、重抓次数、完成时间和失败类型。
- 技能质量：code usage/perplexity、跨任务 purity/NMI、task-ID probe、outcome probe、边界 F1、平均 token 数。
- 因果复用：把同一 `z` 交换到兼容的新任务上下文，测成功/后果一致性；随机或不兼容交换作为负对照。
- 子目标可达性：预测子目标与实际段末状态距离、低层到达率；增加 final-goal oracle 与 oracle intermediate subgoal 两个特权对照。
- 负迁移：与从头/直接微调相比，在含新技能的任务上是否更差。

### 4.4 三个关键消融

1. 固定 8/16/32 帧 vs 事件边界：验证收益来自边界而不是额外参数。
2. 纯离散 vs 纯连续 vs `z+r`：验证共享性与精度的权衡。
3. 动作重建约束 vs outcome/contact 一致性：验证 latent 是否学习可复用技能，而非轨迹压缩。

## 5. 论文贡献建议

将贡献控制在三项，便于实现与答辩：

1. 提出一种由接触/对象后果事件驱动的跨任务技能 tokenizer，使技能具有可变持续时间，并以 outcome/contact 一致性形成可检验的离散语义；不把 contact 表示本身声称为首次贡献。
2. 提出离散共享 mode 与状态条件连续 residual 相结合的层级 VLA，保留精细控制能力。
3. 建立针对精细操作的 few-shot 技能评测协议，以严格对照、因果 skill swap、负迁移和 demos-to-success 曲线证明收益来源。

不建议同时承诺大规模跨本体、真实双臂、触觉、多源互联网视频和长时程 RL。硕士论文的主张应先限定为 **同一本体、多任务、精细操作的共享技能与 few-shot 迁移**；跨本体利用现有 canonical 表示做一个扩展实验即可。

## 6. 分阶段实施计划

### M0：可行性审计（2 周）

- 列出实际 Mobile ALOHA 任务、每任务 episode 数、传感器、控制频率和成功标签。
- 自动统计候选边界信号与跨任务 segment 最近邻，判断数据中是否真的存在重复技能。
- 跑通 π0/π0.5 或 ACT 的直接微调基线。

停止条件：若跨任务没有重复阶段，先重新设计任务/采集数据，不能依靠模型“发现”不存在的共享结构。

### M1：最小技能模型（4-6 周）

- 固定窗口 `z+r` tokenizer + frozen visual encoder + 轻量 action decoder。
- 在 3-5 个模拟任务完成 demo-budget 曲线和纯离散/连续消融。
- 目标是验证共享瓶颈是否存在正迁移，不追求完整 VLA。

### M2：核心创新（6-8 周）

- 加入事件边界、outcome/contact loss、task invariance。
- 接入 π0 action expert，完成核心基线与因果 skill swap。
- 选择最有效的一个机制作为论文主贡献，删除无收益组件。

### M3：真实机器人与论文（6-8 周）

- 2-3 个窄容差真实任务，每个新任务按 demo budget 采集。
- 报告失败模式、置信区间和负迁移，不只给最佳 checkpoint。
- 完成复现实验、消融表、方法图与论文写作。

## 7. 当前风险与应对

| 风险 | 诊断 | 应对 |
|---|---|---|
| codebook collapse | 少数 code 占据绝大多数 segment | FSQ/residual VQ、usage regularization；但不把均匀 usage 当作语义证据 |
| task-ID 泄漏 | `z` 可高精度预测任务但不能跨任务交换 | task adversarial、跨任务 outcome pairing、held-out composition |
| 量化损害精度 | 去 residual 后末端误差/失败显著上升 | 状态条件连续 residual、闭环 receding horizon |
| 边界不稳定 | 不同 seed/扰动下边界漂移 | 事件弱监督、持续时间先验、边界稳定性指标 |
| 无力觉却主张接触技能 | contact label 只能由代理信号猜测 | 限定结论为视觉-本体代理；或增加腕力/电流传感 |
| 数据泄漏 | 同一对象/轨迹变体跨 train/test | 按任务组合、对象实例和场景分组拆分 |
| 算力超限 | 从头训练 VLA 无法充分消融 | 冻结 backbone、LoRA、先小模型验证机制 |

## 8. 待确认的工程约束

以下信息将决定任务范围与模型规模：实际机器人型号与动作空间、GPU 型号/数量、当前任务清单及每任务演示数、是否有腕力矩/触觉/电流、论文剩余时间。未确认前，本报告以 Mobile ALOHA、RGB+proprioception、可选力觉和单机微调为假设。

## 参考资料索引

- 主登记表：`literature_registry.md`
- 潜技能与 skill token：`references_latent_skills.md`
- 层级 VLA：`literature_hierarchical_vla.md`
- 动作 tokenizer 与跨本体：`references_action_representation.md`
- 阶段日志：`progress_log.md`
- 模型详细设计与决策解析：`model_design.md`
