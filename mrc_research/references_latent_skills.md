# 潜变量技能、Skill Token 与时序抽象文献调研

> 调研范围：机器人模仿学习与 VLA 中的潜变量/离散技能发现、skill token、latent plan 和时序抽象。  
> 目标：判断哪些方法最适合作为“跨精细任务共享的隐技能表示”，以提高小样本效率与任务成功率。  
> 状态：本轮调研完成；后续可随新论文和复现实验继续更新。  
> 最后更新：2026-08-13

## 核验口径

- 题名、作者、年份、venue、arXiv ID 以论文原文或官方项目页为准。
- “评测证据”仅记录论文公开报告，不把作者结论当作跨论文可直接比较的统一指标。
- “与课题关系”区分可直接复用的模块、值得验证的假设，以及已知局限。

## 文献条目

### 1. LISA: Learning Interpretable Skill Abstractions from Language

- **作者 / 年份 / venue**：Divyansh Garg, Skanda Vaidyanath, Kuno Kim, Jiaming Song, Stefano Ermon；2022；NeurIPS 2022。
- **原文**：[arXiv:2203.00054](https://arxiv.org/abs/2203.00054)；[项目页](https://div99.github.io/LISA/)。
- **核心方法**：层次化模仿学习。高层 skill predictor 以语言指令与历史为条件，每隔固定的 `H` 步输出一个经 vector quantization 得到的离散 skill code；低层 causal Transformer 以该 code 和短期状态/动作历史为条件预测动作。语言与行为共同约束 code，使离散码具有可解释性和可组合性。
- **评测证据**：在 BabyAI 的 GoToSeq、SynthSeq、BossLevel 以及模拟 LOReL Sawyer 操作上，与参数量匹配的语言条件 Decision Transformer 比较。论文使用 BabyAI 每级百万轨迹中的 0.1%-10%（1k/10k/100k）训练；低数据段 LISA 在全部环境上持平或优于 flat baseline。LOReL 图像观测下成功率约 40%，论文报告为 Lang-DT 的约 2 倍；还展示未见长指令的 skill composition。
- **局限**：机器人证据仅为短时域、模拟 Sawyer，且 LOReL 只有 6 个已见指令作为主表评测；skill 边界是固定 `H`，不是从接触事件中自适应发现；code 的“可解释”主要来自词频/轨迹可视化，不保证跨 embodiment 或跨数据集同码同义；语言标签是必要监督信号之一。
- **对本课题的影响**：可作为“共享技能应同时预测语言语义与动作后果”的直接先例。对精细操作，更合理的改造是把固定 `H` 换成事件/接触感知边界，并要求同一 skill 在多个任务中具有相似的末端位姿增量、力/触觉变化和可达成子目标。LISA 的语言对齐损失可作为主模型的辅助可解释性约束，而不宜单独承担精细控制。
- **代码**：[div99/LISA](https://github.com/div99/LISA)（项目页公开代码链接；复现实验前仍需检查环境版本与数据下载脚本）。

### 2. Behavior Generation with Latent Actions（VQ-BeT）

- **作者 / 年份 / venue**：Seungjae Lee, Yibin Wang, Haritheja Etukuru, H. Jin Kim, Nur Muhammad Mahi Shafiullah, Lerrel Pinto；2024；ICML 2024（PMLR 235:26991-27008）。
- **原文**：[arXiv:2403.03181](https://arxiv.org/abs/2403.03181)；[项目页](https://sjlee.cc/vq-bet/)。
- **核心方法**：先用 hierarchical residual vector quantization 将高维连续 action chunk 编为多个离散 latent action token，再用 causal Transformer 预测 code，并由解码器重建连续动作；保留 residual offset 以减轻量化误差。它解决 BeT 的 k-means 难扩展到长动作块/高维动作、且量化器不可微的问题，重点是多模态行为生成而非语义子任务分段。
- **评测证据**：论文在 7 个环境覆盖模拟操作、自动驾驶和机器人数据，对比 BeT、Diffusion Policy 等；原文报告总体优于这些基线，并宣称相对 Diffusion Policy 推理约 5 倍加速。该结果证明离散 action chunk 可以兼顾多模态和较快闭环推理，但不等价于已发现跨任务语义技能。
- **局限**：latent 的单位是固定长度 action chunk；codebook 首先优化动作重建，可能把绝对机器人坐标、速度尺度、机械臂构型等 nuisance factors 编进 token；没有天然保证跨任务共享、语言对齐或 skill 边界语义；离散到连续重建误差在插接/按压等高精度末端阶段可能被放大。
- **对本课题的影响**：非常适合作为低层执行器和离散化基线。论文创新不应只重复“VQ 动作 token”，而应研究如何把 VQ-BeT 的动作块 token 变成 **任务无关、对象/接触条件化、可变持续时间** 的共享 skill，并对末端误差设置连续 residual controller。可比较单层 VQ、残差 VQ，以及带跨任务一致性正则的 codebook。
- **代码**：[jayLEE0301/vq_bet_official](https://github.com/jayLEE0301/vq_bet_official)（官方）。

### 3. QueST: Self-Supervised Skill Abstractions for Learning Continuous Control

- **作者 / 年份 / venue**：Atharva Mete, Haotian Xue, Albert Wilcox, Yongxin Chen, Animesh Garg；2024；论文公开版本为 arXiv 2024（需在最终引用前再核正式 proceedings）。
- **原文**：[arXiv:2407.15840](https://arxiv.org/abs/2407.15840)；[项目页](https://quest-model.github.io/)。
- **核心方法**：Quantized Skill Transformer。自监督 skill autoencoder 不把整段动作压成单一 latent，而以更大、更灵活的离散 latent sequence 表示动作段；对 latent 施加来自动作时序的 causal inductive bias，使 skill code 不仅能重建动作，还保留顺序和局部因果结构。下游策略预测离散 skill 表示，再由预训练 decoder 生成连续控制。
- **评测证据**：论文与 VQ-BeT、Diffusion Policy、ACT 等比较。项目页报告 LIBERO-90 的相对提升至少 10.3%，在 LIBERO-LONG 的 10 个未见任务 few-shot 设置中相对次优基线提升 24%；MetaWorld 的 45 任务多任务和 5 个 held-out 任务更接近饱和，各方法相近、QueST 略优。重点证据因此来自复杂长程 LIBERO，而不是较简单的 MetaWorld。
- **局限**：仍依赖人为选定动作窗口；自监督动作重建不自动产生人类语义一致的 skill；较大的离散 latent 序列降低了信息瓶颈强度，可能提高重建却削弱高层时序压缩；公开证据以 benchmark 为主，真实精细接触操作和跨机器人迁移证据有限。
- **对本课题的影响**：这是最接近“共享隐技能作为中间表征以做 few-shot 精细任务”的直接基线。值得复用其 causal skill tokenizer，但应新增三项约束：跨任务同类阶段的对比一致性、接触/对象状态后果预测、以及高精度末端阶段的 residual action head。消融应回答性能来自更大 latent 容量还是来自真正可迁移的技能抽象。
- **代码**：[pairlab/QueST](https://github.com/pairlab/QueST)（官方项目页链接）。

### 4. PRISE: LLM-Style Sequence Compression for Learning Temporal Action Abstractions in Control

- **作者 / 年份 / venue**：Ruijie Zheng, Ching-An Cheng, Hal Daumé III, Furong Huang, Andrey Kolobov；2024；ICML 2024。
- **原文**：[arXiv:2402.10450](https://arxiv.org/abs/2402.10450)。
- **核心方法**：Primitive Sequence Encoding。先训练 vector quantizer 把连续动作变成原子离散码，再把多任务演示的码序列当成文本语料，用 byte-pair encoding（BPE）反复合并高频相邻码，形成可变时长、可复用的 skill token；下游 BC 直接预测这些 skill，decoder 展开为动作序列。
- **评测证据**：在 MetaWorld 与 LIBERO 上做多任务及未见任务 few-shot IL。LIBERO 预训练采用 LIBERO-90（90 个任务，每任务 50 条人类遥操作轨迹）；few-shot 在 MetaWorld 5 个和 LIBERO 8 个未见任务上使用 5 条演示，4 个随机种子，每个 checkpoint 40 次 rollout。论文报告 PRISE 相对 ACT、BC 等基线显著提升，并通过仅 VQ、随机合并等消融支持“高频可变时长序列合并”本身有贡献。
- **局限**：BPE 的频率不等于语义或因果价值，容易把数据采集习惯、静止段和控制频率模式合并为“技能”；token 词表随数据和采样频率变化，跨 embodiment 难直接共享；预训练数据必须含有有意义且重复的行为模式，论文也明确指出任意质量离线数据中的模式未必有用；最大 skill 长度仍是超参数。
- **对本课题的影响**：PRISE 给出了低成本、无需边界标签的可变时长强基线，很适合作为硕士课题的 tokenizer baseline。创新点可明确落在“**后果感知 BPE**”：只有当候选码串在多个任务/物体上产生一致的对象状态变化或接触事件时才合并，并对低频但关键的精细技能（对孔、插入、扣合）用成功相关性纠正纯频率偏置。
- **代码**：[FrankZheng2022/PRISE](https://github.com/FrankZheng2022/PRISE)（官方）。

### 5. Skill Transformer: A Monolithic Policy for Mobile Manipulation

- **作者 / 年份 / venue**：Xiaoyu Huang, Dhruv Batra, Akshara Rai, Andrew Szot；2023；arXiv 2023（基于 Habitat rearrangement 工作）。
- **原文**：[arXiv:2308.09873](https://arxiv.org/abs/2308.09873)。
- **核心方法**：端到端 Transformer 同时输出人工定义的高层 skill 类别（如 navigate/pick/place）和 whole-body 低层动作，把技能模块性与单体策略共享上下文结合，减少传统模块切换时的 hand-off error。
- **评测证据**：在具身 rearrangement benchmark 的新场景上，论文报告 hard rearrangement 成功率约为基线 2.5 倍。
- **局限**：skill 是已有轨迹标签/人工语义类，而非无监督发现；主要难点是长程移动操作与模块切换，不是毫米级接触控制；任务类型和 skill vocabulary 已知，不能直接说明隐技能会自然涌现。
- **对本课题的影响**：它提示一个重要结构选择：训练时应保留“skill classification + action generation”的联合损失，并让动作头看到高层 skill 及原始观测，避免离散瓶颈截断精细信息。若用户数据可用自动阶段标签或少量人工标签，可将它作为上界，与纯潜技能方法比较。
- **代码**：未在 arXiv 元数据中发现明确官方仓库；可参考作者/项目生态中的 Habitat-Lab 实现，但不能在报告中声称已有完整官方代码。

### 6. Learning Latent Plans from Play（Play-LMP）

- **作者 / 年份 / venue**：Corey Lynch, Mohi Khansari, Ted Xiao, Vikash Kumar, Jonathan Tompson, Sergey Levine, Pierre Sermanet；2019；CoRL 2019。
- **原文**：[arXiv:1903.01973](https://arxiv.org/abs/1903.01973)；[项目页](https://learning-from-play.github.io/)。
- **核心方法**：经典 latent-plan 先例。用未分段、无任务标签的人类 teleoperation play，训练 conditional VAE：plan recognition 网络从整段轨迹推断连续 latent plan，proposal 网络仅由起点与目标预测其分布，低层 policy 在 plan 条件下复现动作。测试时无需动作标签即可采样达成目标的计划。
- **评测证据**：在模拟桌面环境 18 个用户指定视觉操作任务上，论文报告 play-pretrained 模型显著优于逐任务 expert policy；同收集时间的 play 覆盖约 4 倍交互空间，并出现扰动恢复、retry-until-success 行为。
- **局限**：连续 latent 往往难解释、难建立稳定的离散技能库；训练/测试依赖目标图像，窗口长度预设；主要是单环境模拟证据，缺乏今天标准下的大规模真实机器人验证。
- **对本课题的影响**：它支持“先在丰富、无分段数据上学 task-agnostic behavior prior，再少样本适配”的总体范式。可把 Play-LMP 作为连续 latent baseline，检验离散共享 skill 是否真的带来更好复用，而不是单纯来自预训练数据量。
- **代码 / 数据**：[learning-from-play 项目页](https://learning-from-play.github.io/)公开代码与数据入口。

### 7. Latent Plans for Task-Agnostic Offline Reinforcement Learning（TACO-RL）

- **作者 / 年份 / venue**：Erick Rosete-Beas, Oier Mees, Gabriel Kalweit, Joschka Boedecker, Wolfram Burgard；2022；CoRL 2022。
- **原文**：[arXiv:2209.08959](https://arxiv.org/abs/2209.08959)；[项目页](http://tacorl.cs.uni-freiburg.de/)。
- **核心方法**：低层由模仿学习得到 latent skill/behavior prior，高层用 task-agnostic offline RL 在 latent skill 空间做 goal chaining，从而把数据中未连续出现的片段拼接成新长程方案。
- **评测证据**：模拟与真实机器人实验中报告相对当时 IL/offline-RL 基线数量级的提升；在真实环境以一个多任务视觉运动策略覆盖 25 个操作任务，并展示未见 skill composition。
- **局限**：高层需要离线 RL、奖励/价值学习，工程和稳定性成本高；其主要收益是长程 stitching，不一定提升单一接触阶段的执行精度；latent skill 仍可能是环境特定运动片段。
- **对本课题的影响**：对论文范围控制很关键。硕士课题可先专注“few-shot imitation + 共享低层 skill”，不必一开始加入 offline RL；但 TACO-RL 可作为未来扩展，说明抽取后的 skill library 如何用于长程重组。
- **代码**：[TACO-RL 项目页](http://tacorl.cs.uni-freiburg.de/)提供实现入口。

### 8. STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization

- **作者 / 年份 / venue**：Hao Li, Qi Lv, Rui Shao, Xiang Deng, Yinchuan Li, Jianye Hao, Liqiang Nie；2025；ICML 2025 Spotlight（PMLR 267）。
- **原文**：[arXiv:2506.03863](https://arxiv.org/abs/2506.03863)。
- **核心方法**：针对 VQ skill tokenizer 的 codebook collapse，引入 rotation-augmented residual skill quantization（RaRSQ），用旋转式梯度机制改善 code 分散和量化；随后用 causal skill transformer 自回归建模多级 codebook 间的依赖，再以 action refinement 补偿离散 skill 到连续动作的误差。
- **评测证据**：在 LIBERO 与真实任务上，论文摘要报告约 12% 基线提升。附录最有价值的证据是去掉 action refinement 后，LIBERO-Long 从 88.5% 降到 37.6%（-50.9%），LIBERO-Goal 从 95.0% 降到 54.4%；这直接说明离散 skill code 单独不足以完成精细控制。RaRSQ 在 LIBERO-Long 的最终量化损失相对标准 VQ-VAE 降低约 97%。
- **局限**：2025 年新工作，尚需独立复现；更均衡的 code 使用率不等于更语义、更跨任务；窗口仍固定；复杂增益同时包含 quantizer、causal transformer、refinement 三部分，需严格消融。
- **对本课题的影响**：STAR 是当前最强且最接近的直接竞争工作，用户的贡献必须超出“防 codebook collapse + causal skill composition + residual refinement”。最合适的差异化是事件驱动可变边界、对象后果一致性、触觉/力觉，以及专门的 few-shot 精细操作数据效率曲线。
- **代码**：arXiv 原文未提供可确认的公开仓库链接；应持续检查作者项目页/PMLR supplementary，暂不声称代码可用。

### 9. Subwords as Skills: Tokenization for Sparse-Reward Reinforcement Learning

- **作者 / 年份 / venue**：David Yunis, Justin Jung, Falcon Dai, Matthew Walter；2024；NeurIPS 2024（首版 arXiv 2023）。
- **原文**：[arXiv:2309.04459](https://arxiv.org/abs/2309.04459)。
- **核心方法**：先聚类连续动作，再用 NLP tokenization 合并重复动作子串，形成时序扩展 action；思想与 PRISE 同源但目标是 sparse-reward RL 的探索效率。
- **评测证据**：论文在多个稀疏奖励域中优于 skill-generation baselines，并报告技能生成与在线 rollout 的计算量降低数个数量级。
- **局限**：重点是探索而非模仿学习/视觉操作；动作频率 token 同样可能缺乏语义；机器人精细任务证据不充分。
- **对本课题的影响**：应作为 PRISE 的重要前驱并准确区分贡献，避免把“BPE 动作技能”误当作新的论文创新。用户可借用其计算效率指标，但主评测仍应是成功率和 demos-to-success 曲线。
- **代码**：[dyunis/subwords_as_skills](https://github.com/dyunis/subwords_as_skills)（官方）。

### 10. SkillDiffuser: Interpretable Hierarchical Planning via Skill Abstractions in Diffusion-Based Task Execution

- **作者 / 年份 / venue**：Zhixuan Liang, Yao Mu, Hengbo Ma, Masayoshi Tomizuka, Mingyu Ding, Ping Luo；2024；CVPR 2024。
- **原文**：[arXiv:2312.11598](https://arxiv.org/abs/2312.11598)；[项目页](https://skilldiffuser.github.io/)。
- **核心方法**：高层从视觉和语言学习离散、可解释 skill embedding，以其条件化 diffusion planner 生成多样的 latent state trajectory，再完成任务执行。相当于把 LISA 式语言技能瓶颈与 diffusion trajectory generation 结合。
- **评测证据**：Meta-World MT10 和 LOReL；项目页报告 LOReL 五种语言改写平均成功率比此前 SOTA 高近 10 个百分点，12 个未见组合指令中约为无技能层基线 2 倍、比 LISA 高约 25%。
- **局限**：LOReL 仍是短轨迹模拟 Sawyer；核心是状态轨迹规划而非高频真实精细控制；skill 可解释性依赖语言词云和 code 激活，仍不能保证因果可复用性。
- **对本课题的影响**：适合作为“离散 skill + 生成式低层”的规划类对照。若本课题使用 diffusion/flow action decoder，需要强调低层是闭环 action chunk 且含接触反馈，而非仅生成 latent state trajectory；共享技能评测需加入 held-out task/对象，不只做语言 paraphrase。
- **代码**：[Liang-ZX/skilldiffuser](https://github.com/Liang-ZX/skilldiffuser)（官方）。

### 11. Latent Action Pretraining from Videos（LAPA）

- **作者 / 年份 / venue**：Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, Minjoon Seo；2025；ICLR 2025。
- **原文**：[arXiv:2410.11758](https://arxiv.org/abs/2410.11758)；[项目页](https://latentactionpretraining.github.io/)。
- **核心方法**：先用当前帧和未来帧训练 VQ-VAE inverse/forward visual dynamics，得到无需机器人 action label 的离散 latent action；再让 VLM 根据图像和任务描述做 latent-action next-token prediction；最后用少量带动作机器人轨迹把 latent 映射到真实动作。核心价值是跨 embodiment 的视觉变化接口。
- **评测证据**：项目页显示：在 BridgeV2（WidowX）预训练、Franka 微调的跨 embodiment 设置，LAPA 平均优于使用动作标签预训练的 ActionVLA/OpenVLA；Open-X 设置的 3 个真实任务中 2 个优于 OpenVLA；仅用 Something-Something V2 的 220k 人类视频预训练也产生正迁移。
- **局限**：单/稀疏帧视觉重建 latent 容易编码相机运动、背景和对象外观，而不是可执行技能；latent action 与真实 action 的对应仍需机器人数据；项目页明确称 LAPA 与 OpenVLA 在双臂设置都表现困难；这一路线更像 visual transition token，不等同于时序扩展 skill。
- **对本课题的影响**：LAPA 扩大了可用无标签视频数据，但也提示题目需要区分 **latent action（帧间变化）** 与 **skill（带持续时间和终止条件的闭环策略）**。可将 LAPA tokenizer 用作视觉先验，再通过对象状态 delta、接触事件和机器人动作 decoder 将其“可执行化”。双臂精细操作正是有明确空间的方向。
- **代码**：[LatentActionPretraining/LAPA](https://github.com/LatentActionPretraining/LAPA)（官方）。

### 12. Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos

- **作者 / 年份 / venue**：Yi Chen, Yuying Ge, Weiliang Tang, Yizhuo Li, Yixiao Ge, Mingyu Ding, Ying Shan, Xihui Liu；2025；ICCV 2025（首版 arXiv 2024）。
- **原文**：[arXiv:2412.04445](https://arxiv.org/abs/2412.04445)；[项目页](https://chenyi99.github.io/moto/)。
- **核心方法**：latent motion tokenizer 从相邻视频帧抽取离散 motion tokens，并以当前帧 + token 重建下一帧；Moto-GPT 在视频-指令上做自回归 next-motion-token pretraining；机器人数据阶段用 action query token 共同微调动作预测和 motion-token 预测。
- **评测证据**：项目页在 CALVIN ABC→D 展示未见环境泛化及数据效率曲线：动作标注微调比例越低，相对无 motion-token 预训练版本的差距越大；还在 SIMPLER 展示人类视频预训练迁移，并用轨迹似然区分成功/失败/随机轨迹。
- **局限**：重建相邻帧可能偏向视觉运动而非控制语义；硬件无关是目标但并非完全证明；对力、接触、遮挡下毫米级对准的表达能力未验证；预训练规模与 backbone 也可能贡献增益。
- **对本课题的影响**：适合作为 VLA 层面的强相关工作，说明 latent token 可作为视频语料与机器人动作之间的桥。用户可进一步把 motion token 变成可变时长、对象中心且接触感知的 skill token；评测必须与等数据、等 backbone、无 skill pretraining 的模型公平比较。
- **代码**：[TencentARC/Moto](https://github.com/TencentARC/Moto)（官方）。

## 2026 新近趋势（预印本，作为方向信号而非稳定基线）

### 13. Skill-Aware Diffusion for Generalizable Robotic Manipulation（SADiff）

- **作者 / 年份**：Aoshen Huang, Jiaming Chen, Jiyu Cheng, Ran Song, Wei Pan, Wei Zhang；2026 预印本。
- **原文**：[arXiv:2601.11266](https://arxiv.org/abs/2601.11266)；[项目页](https://sites.google.com/view/sa-diff)。
- **方法与证据**：用可学习 skill token 表示人工/任务级同类技能，条件化 diffusion 生成 object-centric motion flow，并用 skill retrieval 把 2D flow 转为 3D action；引入 IsaacSkill 并报告模拟、真实与 sim-to-real 泛化。
- **局限 / 课题关系**：它不是无监督 skill discovery，但“对象中心 motion flow + skill prior”正面命中跨任务共享运动。用户必须与这种显式 skill-class 条件方法比较，证明潜技能在没有 skill 标签或新组合上更省样本。

### 14. SA-VLA: State-aware tokenizer for improving Vision-Language-Action Models' performance

- **作者 / 年份**：Tengyue Jiang, Chunpu Xu, Jiayue Kang, Yao Mu；2026 预印本。
- **原文**：[arXiv:2606.30113](https://arxiv.org/abs/2606.30113)。
- **方法与证据**：离散 code 不再对应固定动作原型，decoder 以本体状态 cross-attention 或轻量 adapter 调制同一 token 的连续含义。在 12 个 RoboTwin 任务中报告平均成功率由最强 tokenizer baseline 的 0.29 提升到 0.56，3 个 zero-shot sim-to-real 任务由 0.15 提升到 0.33。
- **局限 / 课题关系**：截至当前日期为非常新的预印本，需谨慎看待；但它揭示本课题的必要条件：所谓“共享 skill token”必须允许状态条件化实例化，否则相同 token 在不同关节构型/接触状态下无法保证精度。建议 decoder 显式输入 proprioception、对象相对位姿和力觉。

> 注：2026 年 7 月后还出现 DLAM、WALA、LMP-tokenizer 等稿件；由于发布时间极新、缺少充分复现与稳定 proceedings，本档案暂不把其结果作为论文设计的主要证据。它们共同指向三个趋势：latent action 的时序组合约束、视觉变化与可执行动作的联合监督、以及可变长度 tokenizer。

## 必要经典基线（用于定位创新）

### 15. CompILE: Compositional Imitation Learning and Execution

- **作者 / 年份 / venue**：Thomas Kipf, Yujia Li, Hanjun Dai, Vinicius Zambaldi, Alvaro Sanchez-Gonzalez, Edward Grefenstette, Pushmeet Kohli, Peter Battaglia；2019；ICML 2019。
- **原文**：[arXiv:1812.01483](https://arxiv.org/abs/1812.01483)。
- **核心方法 / 证据**：可微分、无监督地把演示分成可变长片段并学习事件 latent，能重组到更长序列；在 2D 多任务与连续控制任务中找出边界和事件码，并帮助稀疏奖励层次策略。
- **局限 / 课题关系**：通常要给出每条序列的最大/预期段数，对段数误设敏感，视觉真实机器人证据弱。它是“事件边界可学习”的经典源头，应成为用户事件驱动 tokenizer 的概念基线，但实现可采用现代 differentiable boundary detector，避免照搬其强先验。
- **代码**：[google-deepmind/compILE](https://github.com/google-deepmind/compILE)（官方 DeepMind 仓库）。

### 16. OPAL: Offline Primitive Discovery for Accelerating Offline Reinforcement Learning

- **作者 / 年份 / venue**：Anurag Ajay, Aviral Kumar, Pulkit Agrawal, Sergey Levine, Ofir Nachum；2021；ICLR 2021。
- **原文**：[arXiv:2010.13611](https://arxiv.org/abs/2010.13611)；[项目页](https://sites.google.com/view/opal-iclr)。
- **核心方法 / 证据**：从无定向离线数据学习固定时长连续 primitive latent 与 behavior prior；下游在 primitive 空间学习，以缩短 horizon 并约束策略留在数据支持内。论文覆盖 offline RL、few-shot IL、在线探索/迁移并报告收益。
- **局限 / 课题关系**：固定窗口、连续且难解释，主要 benchmark 非视觉精细操作；但它是 QueST/PRISE 等必须比较的连续 latent 先例。对用户而言，OPAL 代表一个重要反事实：如果连续 latent 已能达到相同样本效率，离散 token 的价值必须由可复用性、组合性或解释性来证明。
- **代码**：[项目页](https://sites.google.com/view/opal-iclr)提供实现/可视化入口。

## 综合结论

- 当前四篇论文已显示：**动作压缩本身不等于共享技能发现**。VQ-BeT 强在多模态动作建模，LISA 强在语言可解释性，QueST 强在可迁移的结构化潜表示，PRISE 强在可变时长压缩。用户课题应把三者组合为一个可证伪的问题：加入“跨任务一致后果/接触”的归纳偏置后，skill token 是否比纯动作重建或纯出现频率 token 在少样本精细任务上更有效？
- 精细任务的关键风险是量化误差。合理结构是高层预测离散共享 skill，低层 decoder 生成 action chunk，同时用连续 residual head 或闭环视觉/力觉纠偏；不能仅以 token prediction accuracy 作为成功代理。
- **文献缺口已经较清楚**：现有方法分别解决语言对齐、动作压缩、codebook collapse、时序组合或视频预训练，但很少同时要求 latent skill 对“对象状态后果/接触模式”跨任务保持不变，并在真实或双臂精细任务上以少样本成功率验证。这比笼统的“skill-learning VLA”更可辩护。
- **严谨定义**：本课题中的 skill 不应只是连续 `K` 帧动作的别名。建议定义为 `z=(mode, termination)`：给定对象中心状态和接触上下文，执行后产生可预测的语义/几何状态变化；持续时间由终止事件决定，而连续 residual 负责实例级精度。
- **避免过度宣称**：LIBERO/MetaWorld 上平均成功率提升只能支持模拟多任务迁移；若没有真实插接、装配或双臂实验，不应把结论写成“提高精细任务成功率”。至少要加入一个含窄容差或接触的真实任务，或明确限定为模拟验证。

### 阅读与复现优先级

| 优先级 | 工作 | 原因 |
|---|---|---|
| P0 | QueST、STAR、PRISE | 最接近课题主张，分别覆盖 transferable skill、量化/精修、可变时长；必须复现或至少在统一数据上重实现 |
| P0 | ACT/Diffusion Policy/flat Transformer | 验证收益是否真的来自 skill，而不是更强 decoder/backbone；必须做等参数和等数据比较 |
| P1 | LISA、SkillDiffuser | 提供语言对齐、组合性与解释性对照 |
| P1 | VQ-BeT、OPAL | 提供离散 action chunk 与连续 latent 两类关键反事实 |
| P1 | LAPA、Moto | 若论文包含 VLA/无标签视频预训练，则是必须覆盖的最新背景；若只做机器人演示，可作为扩展 |
| P2 | Play-LMP、TACO-RL、CompILE、Subwords as Skills | 用于理论脉络、边界学习和长程技能重组，不必全部工程复现 |

### 一句话研究问题

> 在多任务精细操作演示中，能否通过事件驱动分段和跨任务“对象后果 + 接触模式”一致性，学习可变时长离散技能；并通过状态条件连续精修，使新任务仅需少量演示即可可靠组合这些技能？

## 建议实验路线

### 推荐主线：Outcome- and Contact-Consistent Skill Tokenizer（暂名）

1. **数据单位**：将多任务演示变为对象中心表示：末端相对对象的 SE(3)、gripper、proprioception、可选 wrist RGB/力矩。避免 token 只记住绝对工作空间位置。
2. **候选边界**：由视觉对象状态突变、gripper 开闭、接触/力变化和速度驻点产生软边界；训练 differentiable boundary head，并约束合理的平均时长/边界稀疏性。
3. **技能编码**：每段用 residual VQ 或 FSQ 编离散 `z`；decoder 以 `z + 当前状态 + 对象关系` 生成动作块。沿用 STAR 的状态/动作 refinement，避免量化精度损失。
4. **共享约束**：对跨任务中具有相同对象后果的片段做对比拉近；对动作相似但后果不同或接触模式不同的片段推远。辅助预测段末对象状态 delta、contact class、termination probability。
5. **高层策略**：VLM/轻量 Transformer 输入视觉与语言，预测 `z` 及终止；低层 skill decoder 冻结或小学习率适配。少样本新任务只训练高层/adapter，测试共享技能带来的数据效率。

### 最小可发表实验矩阵

| 问题 | 对照 | 指标 |
|---|---|---|
| skill bottleneck 是否有效 | ACT/Diffusion Policy/等参数 flat policy | task success，最终位姿/插入深度，失败类型 |
| 离散是否优于连续 latent | OPAL/Play-LMP 风格连续 latent | 1/5/10/20 demos success 曲线，latent probing |
| 可变边界是否有效 | 固定 8/16/32 steps（QueST/STAR 风格）与 PRISE-BPE | success，平均 token 数，boundary F1（若有少量标注） |
| outcome/contact 约束是否有效 | 仅动作重建 VQ；仅语言对齐 LISA | held-out task/object success，跨任务 token purity |
| 精度修正是否必要 | 去掉 continuous residual/refinement | 末端误差、接触失败率、success（预期应复现 STAR 的明显退化趋势） |
| 是否真正省样本 | 从头训练与全量微调 | 达到固定成功率所需 demo 数、AUC over demo budgets |

### 数据与范围建议

- **首选可控模拟**：LIBERO-90 预训练 + LIBERO-LONG/自定义 held-out compositional tasks；为了“精细”主张，增加 MimicGen/robosuite 中 nut assembly、peg insertion 或 tool hanging 等窄容差任务，而不是只做 pick-place。
- **真实验证**：选 2-3 个共享阶段明显的任务，例如 grasp-and-align、插接、扣合/按压；每个新任务 5/10/20 条演示。若使用 Mobile ALOHA 数据，可把双臂 `approach → grasp → align → insert/transfer → release` 作为跨任务候选阶段。
- **算力控制**：先冻结视觉 encoder，用已有 ACT 或 Diffusion Policy action decoder 证明 tokenizer 假设；不建议硕士初期从头训练大 VLA。待 skill 机制有效后，再接 OpenVLA/Octo/轻量 VLM 做语言泛化。

### 论文创新边界

- 不能把“VQ 动作 token”作为创新：LISA、VQ-BeT、QueST、STAR 已覆盖。
- 不能把“BPE 得到可变长技能”作为创新：Subwords as Skills 和 PRISE 已覆盖。
- 不能把“视频 latent action 预训练”作为创新：LAPA、Moto 已覆盖。
- 可以形成清晰贡献的是：**事件驱动的可变时长分段 + 跨任务对象后果/接触一致的离散技能 + 状态条件连续精修 + 在窄容差 few-shot 操作上的系统验证**。

## 来源核验说明

- 以上题名、作者、arXiv ID 和 venue 主要由 arXiv 官方 API、论文 HTML/PDF 与官方项目页交叉核对。
- 实验百分比均标明来自论文/项目页；不同 benchmark、rollout 数和数据预算不可横向直接排序。
- 2026 条目均按预印本处理，不把其宣称视为已独立复现事实。
