# Skill-Learning VLA 硕士课题调研进度日志

> 本文件记录阶段性事实、判断变化、决策与待验证问题。时间均为 Asia/Shanghai。

## 2026-08-13 阶段 0：目标与现有基础

### 研究目标

构建一个面向精细机器人操作、具有高样本效率的 skill-learning VLA。核心设想是从多任务轨迹中提取可复用的共通技能作为隐变量，使新精细任务能通过技能组合或适配，以更少示教获得更高成功率。

### 已确认的工程基础

- `openpi` 仓库已存在，版本提交为 `15a9616`（2026-06-16），包含 π0、π0.5、π0-FAST、LeRobot 数据加载、ALOHA/DROID/LIBERO 配置及 LoRA 微调入口。
- `vla-data-pipeline` 已能将 Mobile ALOHA HDF5 转换为 LeRobot v3.0，并包含数据清洗与检查。
- 数据管线定义了跨本体 canonical state/action：固定 128 维并配套布尔 mask；动作中包含关节、末端位置/旋转、夹爪和双臂槽位。
- 该 canonical 表示为跨本体技能学习提供了接口，但当前尚未定义技能分段、技能标签或潜技能字段。

### 初步研究判断

1. 论文不能只把“动作 chunk 压缩”称为技能学习。需要明确区分：语言子任务、离散技能 token、连续时序 latent、动作 tokenizer/chunk。
2. 关键科学问题应是：在控制任务、数据规模和参数量后，共享技能瓶颈是否比直接 VLA 微调显著提高 held-out 精细任务的样本效率与成功率。
3. 精细任务要求技能 latent 保留接触阶段、力度/夹爪事件和微小位姿修正；过强离散压缩可能损害精度。
4. 必须设置负迁移与技能可辨识性实验，否则无法证明 latent 是“共通技能”而非任务 ID 或轨迹压缩码。

### 当前风险

- 用户尚未明确实际机器人、GPU、可采集示教数和任务清单；方案先以 Mobile ALOHA/双臂精细操作为主要假设，后续需用工程约束收敛。
- 128 维 canonical 动作与 openpi 默认 32 维动作存在接口差异，需要研究时决定是在数据变换层选择有效槽位，还是改造模型动作头。
- 公共数据大多缺少力觉；若论文主张“精细接触”，仅靠 RGB+本体状态可能构成能力上限。

### 下一阶段

- 并行建立四条文献证据链：潜技能发现、层级 VLA/规划、动作表示/时序抽象、精细操作与样本效率评测。
- 为每篇核心工作记录：准确出处、问题设定、方法、数据/指标、局限、代码，以及对本课题的直接影响。
- 基于证据选择一个主方案和一个风险更低的最小可行版本。

## 2026-08-13 阶段 1：研究版图与创新边界初筛

### 文献检索组织

- 已建立三个主题文件并采用独立写入，防止并行调研互相覆盖：潜技能/skill token、层级 VLA、动作 tokenizer/跨本体表示。
- 来源优先级为论文原文与作者项目页，其次为会议元数据；已避免将搜索引擎摘要直接作为结论证据。
- 第一批确认的核心邻近工作包括 LISA、VQ-BeT、SkillMimicGen、HiRT、TACO、UniAct、FAST；仍在核对 PRISE、QueST、LOTUS、SkillPlug 等工作。

### 术语边界

当前把相关方法分为四层：

1. `task plan`：语言或视觉子目标序列，解决长时程规划，不直接保证精细控制。
2. `semantic skill`：可跨任务复用、具有 initiation/termination 和子目标语义的时序抽象。
3. `latent action`：对动作 chunk 的离散或连续压缩，可能提高生成效率，但未必具有技能语义。
4. `motor primitive`：短时域闭环控制原语，强调接触、顺应性和低层精度。

本课题的创新应位于第 2 与第 4 层之间：高层 latent 需要可复用，低层执行仍应连续、闭环且保留精细残差。单纯替换 FAST tokenizer 或增加 VQ codebook 不足以形成有说服力的论文贡献。

### 主方案雏形

暂定工作名为 **CASK（Contact-Aware Shared Skills）**：

- 用视觉、本体状态和动作变化联合发现边界；有力/触觉时加入接触变化，没有时用夹爪事件、速度骤变、运动停顿和视觉交互作为代理。
- 每个 segment 编成“离散共享技能 ID + 连续实例 residual”。离散 ID 表示跨任务共通结构，residual 保存物体几何、位姿和接触细节。
- 高层 VLA 根据图像、语言和历史预测技能及终止条件；低层 π0 flow head 或 diffusion policy 根据技能 latent 生成短动作 chunk。
- 用任务对抗或信息瓶颈抑制 skill code 泄漏完整 task ID，同时用跨任务对比学习把具有相同前后状态变化的 segment 拉近。

### 尚未通过证据确认的假设

- 动态边界是否在精细任务中稳定优于固定 chunk。
- 离散 ID 加连续 residual 是否比纯连续 latent、纯离散 VQ 和 FAST action token 更好。
- 现有 Mobile ALOHA 数据是否包含足够多跨任务重复原语；若没有，共享技能无法被可靠识别。
- canonical 128 维动作是否应直接进入模型，还是仅用于 skill encoder 的跨本体对齐，执行器仍使用本体原生动作。

## 2026-08-13 阶段 2：潜技能证据链完成与方案收敛

### 已完成

- 潜技能主题第一轮完成，共核验 16 篇核心/经典工作，并记录代码、证据、局限、复现优先级与实验矩阵。
- 已建立工作报告 `research_report.md` v0.1，形成问题定义、CASK 方法、实验协议、工程接入和阶段计划。
- 当前工作区未发现实际机器人数据文件，仅有转换/清洗代码和模型仓库；因此尚不能对任务重叠度、候选技能频率或数据规模作实证判断。

### 关键决策

1. 主论文先限定为同一本体多任务 few-shot 精细操作；跨本体作为扩展，避免范围失控。
2. canonical 128 维表示用于 skill encoder 的共享对齐；低层执行先保留原生动作维。
3. 采用离散共享 mode + 连续 residual + termination，禁止用纯离散 token 承担毫米级控制。
4. 最主要的创新证据是 outcome/contact 一致性和事件边界带来的 held-out task 样本效率，而不是 codebook usage 或可视化。

### 下一验证门槛

- 获取真实数据根目录后统计 task、episode、FPS、传感器和动作空间。
- 检查预训练任务与 held-out 任务是否含可复用阶段；若缺乏重叠，需先调整任务设计或采集协议。
- 确定 GPU 与论文时间，决定使用 π0 LoRA、冻结 backbone，还是先用 ACT/Diffusion Policy 验证技能机制。

### 证据质量修正

- 并行下载阶段发现 `paper_2305.07703` 实际为无关的理论物理论文（The Hydrohedron），说明单靠猜测 arXiv ID 不可靠；该文件不进入文献证据，相关引用必须重新由题名/项目页核验。
- 其余缓存论文也只能在题名与正文匹配后使用。总报告只采用已确认来源，不因“最新”而降低核验标准。

### 新增直接竞争：SkillVLA（2026 预印本）

- 论文研究真实双臂中的组合多样性，将左右臂单臂技能复用与真正协作技能分开建模。
- 论文报告 π0.5 与 π0-FAST 在未见左右臂组合上为 0% 成功，SkillVLA 达到 51%；在 LIFT BAG continual setting 中，5 条新双臂示教已接近峰值。
- 对本课题的影响：若以双臂技能复用为主张，不能只提出 per-arm decoder 或协作 gate。创新必须强调无人工技能库的事件分段、跨任务 outcome/contact 一致性与精细 residual，并设置 SkillVLA 风格组合拆分作为强基线。

## 2026-08-13 阶段 3：层级 VLA 证据链完成

### 新结论

- 最强 few-shot latent skill 证据是 QueST：LIBERO-LONG 5-shot 68.8%，高于次优约 14 个百分点；PRISE 报告 5-shot +9.2 个百分点。
- SuSIE 提供层级提高精细度的直接证据：中间视觉子目标甚至超过“真实最终目标图”特权基线，真机平均 0.78->0.88、CALVIN 0.66->0.95。
- RT-H 总体相对 flat RT-2 提高约 15%，但在最精细罐操作上动作 K-means 簇反而更好；语言子任务会丢失接触几何与速度细节。
- Moto 在仅用 1% action-labeled data 时报告 52.5%，从头训练为 0%，但它依赖大规模视频预训练，故只能称“下游动作标签高效”，不能称总体少数据。

### 架构修订

CASK 从两层修订为三层：

1. 语言子任务：回答“做什么”，提供可解释规划与干预接口。
2. 视觉/几何子目标：回答“到哪里/形成什么对象关系”，用可达性约束连接规划和控制。
3. 接触感知 latent skill：回答“怎么做”，以离散共享 mode + 连续 residual 执行精细动作。

最小实现仍优先完成第 2、3 层；语言 planner 可复用已有 VLM 或数据中的 task instruction，避免早期工程范围膨胀。

## 2026-08-13 阶段 4：动作表示与跨本体证据链完成

### 新增强竞争与反证

- UniVLA 已覆盖 task-centric 离散 latent action + 跨本体视频预训练 + 少量动作 decoder；CASK 的最大重合风险是“仅把固定 latent action 换一个名字”。
- CLAM 表明连续 latent action 可能更适合精细控制；本课题必须比较 continuous only、discrete only、discrete+residual。
- FAST 解决高效动作 tokenization，ACT/Diffusion Policy 分别是固定 chunk 与连续多峰 chunk 的强基线。
- CrossFormer 使用共享骨干但本体专用动作头，且未观察到显著跨本体正迁移；这支持 canonical 表示先用于 skill encoder，而不强制统一低层动作头。
- UHAS/Cloak 说明跨本体收益可能来自几何/视觉对齐，构成 skill 解释之外的替代假设。

### 第一轮调研收敛判断

当前可成立的窄而强创新为：**接触/outcome 事件定义的可变时长边界 + 跨任务后果/接触一致的离散 mode + 状态条件连续 residual + π0 闭环低层执行**。任何一个更宽泛的表述，如“skill token”“跨本体统一动作”“可变长 tokenizer”“双臂技能复用”，都已有直接近邻工作。

### 最终措辞约束

- 不声称首次 universal action / embodiment decoder：CVPR 2025 UniAct 已覆盖。
- 不声称首次 cross-embodiment latent：LAD、OPFA、UHAS 等已覆盖。
- 不声称首次 contact consistency / contact transfer：2026《Transferring Contact, Not Just Motion》已覆盖 calibrated contact interface。
- 可主张的组合创新是：**接触/后果事件用于可变时长边界；跨任务 outcome/contact 一致性用于形成技能语义；连续 residual 用于保留窄容差精度；在 held-out few-shot 精细任务中做因果验证。**

## 2026-08-13 第一轮调研交付

- 三个文献主题均完成第一轮：潜技能 16 篇核心/经典工作，层级 VLA 11 篇核心工作，动作表示覆盖 FAST/ACT/DP/UniVLA/UniAct/CLAM/CrossFormer/LAD/OPFA/UHAS/Cloak/contact transfer 等。
- `research_report.md` 已汇总为结构化方案，包含研究问题、三层架构、openpi 接入点、基线、消融、指标、风险与实施计划。
- 所有 Markdown 已按严格 UTF-8 解码检查，无替换字符；文献检索临时 PDF/HTML/JSON 已清理，只保留报告文件。
- 第一轮尚未执行模型训练或真实数据统计，因为工作区未包含数据集本体，也未提供 GPU/任务/传感器约束。下一轮从数据可行性审计开始。

## 2026-08-14 阶段 5：CASK 模型规格细化

### 本阶段产物

- 新增 `model_design.md`，将报告中的概念方案展开为输入接口、边界/终止、混合 tokenizer、在线 prior、grounded subgoal、π0 接入、损失、训练协议、评测和停止门槛。
- `research_report.md` 更新为 v0.2，并将详细规格作为后续实现的模型来源。

### 关键补全

1. 明确训练轨迹后验 `q(z,r|segment)` 与部署在线先验 `p(z,r|context)`。主实验必须使用 prior-only rollout，防止从完整未来动作段编码 latent 的信息泄漏。
2. 将离线双向 boundary posterior 与在线因果 termination head 分开；前者只生成训练分段，后者才控制真实执行。
3. 初始离散实现选择 4 维、每维 3 级的 FSQ，连续 residual 初始为 16 维；这只是 M1 起点，必须以 continuous-only 与 EMA-VQ 对照决定是否保留。
4. posterior tokenizer 禁止输入语言、task ID 和绝对工位；对象中心 outcome/contact 约束定义共享语义，绝对坐标只进入 residual prior/低层控制。
5. canonical 128 维表示仅进入技能编码分支，π0 flow loss 保持在本体原生有效动作维，mask 同时进入 attention 和 loss。

### 设计理由与证伪

- 使用 `z+r` 是为同时满足跨任务复用和窄容差精度；若 continuous-only 在统一预算下全面更优，则离散共享 mode 的必要性不成立。
- 使用事件边界是因为技能以接触/后果终止定义；若其稳定性和 rollout 都不优于固定窗，则不再声称边界创新。
- 语言 planner 初期冻结，是为了隔离技能机制收益；只有 L2/L3 已成立后才扩展端到端语言规划。
- skill swap 采用“交换 `z`、在目标上下文重算 `r`”，避免把对象特定轨迹整体复制后误判为共享技能。

### 尚未解决

- 当前 `mrc_research` 目录仍无实际数据或 GPU/机器人配置，因此尚不能锁定帧率、时长、模型宽度和传感器分支，也不能执行 M0/G0 数据可行性审计。代码位于相邻目录：`../openpi` 已核验 commit `15a9616a00943ada6c20a0f158e3adb39df2ccac` 及 `Pi0.embed_suffix`；`../vla-data-pipeline` 已核验 commit `31760a3ad4d9aba845fd73237a767b56fa5cace7` 及 canonical mask 写入接口。两者不是本报告目录内文件。

### 独立审查后的修订

- 三个子智能体分别从技术可实现性、证据一致性和规格完整性审查 v0.1；主线未被否定，但发现训练/部署闭环与归因控制不足。
- `model_design.md` 更新为 v0.2：补充 M1 conditional flow decoder，使 `r` 获得明确动作学习信号；规定 posterior teacher forcing 到 100% predicted 条件的 curriculum，并明确 stop-gradient。
- 默认边界流程改为冻结/交替重分段，不再声称 action loss 可穿过 hard NMS；离线双向与在线因果塔只共享逐帧投影。
- 离散/实例输入拆为 `x^z/x^r`，目标、绝对关节和工位禁止进入 `z`；mask 拆为时间、feature、sensor 三类。
- 主评测改为完整因果链：预测 `g,z,r` 与在线 termination；增加 subgoal×skill 2×2、事件切分但无 skill、PRISE/BPE、UniVLA、QueST/STAR、CLAM/OPAL 等归因对照。
- Gate 改为相对预注册最强主基线、下游任务收益和条件 task probe；具体数值门槛将在 M0 pilot 后、最终测试前写入带哈希协议，避免当前无数据时伪造阈值。

### 完成审计

- 三轮技术复核后，最终审查结论为“无剩余实现阻断”。最后补齐了 openpi 32 维动作在训练/采样 ODE 每一步的无效槽 mask、可执行的多候选 subgoal proposal、空正对 SupCon 处理、可变时长 action padding，以及正常终止/删失监督映射。
- `model_design.md` 当前 SHA-256：`3D34D5084F7B17AD2A966A7F8532D5882BC88B2C66C43B5E36AE41D058452D96`；`research_report.md` 当前 SHA-256：`4958AF70304863A47F0576CA5840896E9FC923B036A6F122874CA1C018CA3D50`。
- 格式检查通过：详细设计的展示公式分隔符成对、代码围栏成对、UTF-8 无替换字符；报告与日志已建立双向版本/产物记录。
- 外部接口复核：`../openpi` commit `15a9616...` 的 `pi0_config.py` 明确 `action_dim=32`，`pi0.py` 当前在 action 维直接求 mean，因此设计中要求在 reduction 和采样 ODE 前后显式应用 `M_π`；数据管线 commit `31760a3...` 明确保存 dataset-constant boolean `action_canonical_mask`。
- 本阶段完成的是模型与实验的实现级预规格，并未声称训练/真实机器人结果已经完成。下一阶段入口仍是 M0/G0：获得数据和资源约束后冻结 `experiment_protocol.yaml`，再实现与训练。
