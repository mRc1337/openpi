# 层级 VLA、子任务规划与长时程技能策略文献档案（2023-2026）

> 调研范围：层级 Vision-Language-Action（VLA）、语言/视觉子任务规划、长时程 manipulation、skill-conditioned policy；重点判断其是否真正提高低样本精细操作成功率。
>
> 术语边界：**语言子任务**是人可读的中间指令；**视觉子目标**是目标图像/视频/空间表征；**latent skill** 是从动作或轨迹中自监督学习的离散/连续隐变量。三者都可作为时间抽象，但信息瓶颈、监督来源和控制作用完全不同，不能混称为 skill。

## 阶段记录

- 2026-08-13 00:50：建立“任务分解表征 -> 技能/控制器 -> 长时程闭环 -> 低样本精细迁移”证据链，优先论文 PDF、作者项目页和官方代码。
- 2026-08-13 01:10：确认 RT-H 属于语言动作层级，不是 latent skill；其精细任务消融反而提示动作聚类有局部优势。
- 2026-08-13 01:35：完成 SuSIE、PRISE、VQ-BeT、QueST、Moto 正文实验核验。视觉子目标和 latent action 都有直接精度/低样本证据，但评测域差别很大。
- 2026-08-13 02:00：完成 Hi Robot、π0.5、SkillVLA 和 GPLA 核验。语言层级对长时程和组合泛化有强证据，但精细操作增益通常来自低层策略或视觉/动作表征。
- 2026-08-13 02:25：完成综合与课题落地建议。核心结论是采用“双层 skill”：语言子任务负责可解释规划，latent skill 负责可复用精细动作，并以视觉/几何子目标连接二者。

## 一、直接相关文献卡片

### 1. RT-H: Action Hierarchies Using Language

- **作者 / 年份 / 载体**：Suneel Belkhale, Tianli Ding, Ted Xiao, Pierre Sermanet, Quan Vuong, Jonathan Tompson, Yevgen Chebotar, Debidatta Dwibedi, Dorsa Sadigh；2024；arXiv:2403.01823（未从论文原文核实正式 venue）。[论文](https://arxiv.org/abs/2403.01823)；[项目页](https://rt-hierarchy.github.io/)
- **skill 类型**：**语言子任务/语言动作**，不是 latent skill。中间短语如 “move arm forward”“grasp the can”，由同一 VLM 先预测语言动作，再以任务、视觉和语言动作预测低层动作。
- **数据与评测**：Diverse+Kitchen 共 100k demonstrations，其中 Kitchen 70k、6 类，Diverse 30k、24+ 类；8 个较难真机任务，每方法 80 次试验。
- **关键结果**：平均成功率比平坦 RT-2 高 15%，6/8 任务更好；动作验证 MSE 约低 20%。语言动作换成 one-hot 明显下降；直接 K-means 动作簇总体略低，但在开/关开心果罐等最精细任务上反而超过语言层级。语言干预可接近满成功率，学习语言干预相对 IWR 提升 50%。
- **局限**：中间标签由自动流程得到但仍依赖语言语义；两次串行查询增加延迟；没有 few-shot 新精细任务评测。最重要的是，语言表示会丢失接触几何和力信息，论文自身的精细任务结果已经暴露这一点。
- **对课题影响｜核心继承 + 风险提示**：继承“显式层级 + 上下文条件低层控制”，但不要把语言动作直接当作最终的共通隐技能。应将语言层用于规划/干预，将动作 latent 用于精细控制；RT-H、RT-H-Cluster、flat VLA 是必须做的三组对照。

### 2. Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models（SuSIE）

- **作者 / 年份 / 载体**：Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya, Homer Walke, Chelsea Finn, Aviral Kumar, Sergey Levine；2023 arXiv，ICLR 2024。[论文](https://arxiv.org/abs/2310.10639)；[项目页](https://rail-berkeley.github.io/susie/)；[官方代码](https://github.com/kvablack/susie)
- **skill 类型**：**视觉子目标层级**，不是 latent skill。InstructPix2Pix 根据当前图像和语言生成可达的未来图像，语言无关的 goal-conditioned policy 执行，再闭环重规划。
- **数据与评测**：BridgeData V2 60k+ 轨迹（45k 语言标注、15k action-only），另用约 75k 筛选后的 Something-Something 人类视频；CALVIN ABC->D 零样本；WidowX 真机 3 场景 9 任务，每任务 10 次。
- **关键结果**：CALVIN 连续完成 1/2/3/4/5 条指令成功率 0.87/0.69/0.49/0.38/0.26，显著高于当时基线。真机三场景平均为 0.87/0.50/0.88；Scene B 精细抓取中 RT-2-X 为 0，SuSIE 为 0.50。更关键的特权对照中，Oracle GCBC 使用真实最终目标图仍被 SuSIE 超过：真机平均 0.78->0.88，CALVIN 平均 0.66->0.95，说明中间视觉子目标确实改善精度，而不仅是语义识别。
- **局限**：子目标生成器和低层 policy 分开训练，不知道低层可达性；图像生成可能幻觉；最终仍受低层 policy 限制；没有 few-shot 目标任务适配曲线。
- **对课题影响｜核心继承 + 评测借鉴**：这是“层级提高精细成功率”最直接证据。可继承“短时可达视觉子目标”，并让 latent skill decoder 同时以视觉子目标为条件。评测必须加入 final-goal oracle，排除只是目标表达更好的混淆因素。

### 3. PlayFusion: Skill Acquisition via Diffusion from Language-Annotated Play

- **作者 / 年份 / venue**：Lili Chen, Shikhar Bahl, Deepak Pathak；CoRL 2023。[论文](https://arxiv.org/abs/2312.04549)；[项目页](https://play-fusion.github.io/)
- **skill 类型**：语言条件下的**离散瓶颈/隐式技能表征**。同时量化 U-Net 行为特征和语言 embedding；它不是显式可调用的长时 skill token 库，更像鼓励技能聚类的 latent bottleneck。
- **数据与评测**：语言 hindsight 标注的非最优 play；CALVIN、Franka Kitchen、Ravens，以及 3 个真机场景。每个真机场景约 250 episode（约 15 小时），通过语言改写增广至 750-1000 条。
- **关键结果**：CALVIN A/B 成功率 45.2/58.7，明显超过 C-BeT 的 26.3/23.4；真机 Dining/Cooking/Sink 为 45/30/20%，基线最高 20/0/10%。还评测 A+B、C+D 训练后 A+D 的技能组合泛化。去除离散瓶颈平均会下降。
- **局限**：仍需要人工 hindsight 语言标注；真机绝对成功率不高；没有严格 few-shot 精细适配；“skill”语义性主要通过 embedding 和消融间接证明。
- **对课题影响｜关键对照 + 外围背景**：适合作为“扩散 policy + latent bottleneck”对照，也提示可利用非专家 play 降低数据成本。但若论文主张提取共通 skill，需要比 PlayFusion 更强的可复用性判据：跨任务检索、组合、冻结 decoder 的适配和技能占用统计。

### 4. PRISE: LLM-Style Sequence Compression for Learning Temporal Action Abstractions in Control

- **作者 / 年份 / venue**：Ruijie Zheng, Ching-An Cheng, Hal Daume III, Furong Huang, Andrey Kolobov；ICML 2024（PMLR 235）。[论文](https://arxiv.org/abs/2402.10450)；[官方代码](https://github.com/FrankZheng2022/PRISE)
- **skill 类型**：明确的**离散 latent skill token**。先把 state-action 量化成逐步 code，再对 code 序列做 BPE，获得可变时长、可解码为闭环低层动作序列的 temporal skill。
- **数据与评测**：LIBERO-90 每任务 50 条人类 demonstration；MetaWorld 45 个预训练任务每任务 100 条脚本轨迹。下游在 8 个 LIBERO-LONG 未见任务和 5 个 MetaWorld 难任务上做 5-shot IL。
- **关键结果**：LIBERO 5-shot 相对最佳基线绝对提升 9.2%；MetaWorld 大幅超过 BC、ACT 和去 BPE 版本。BPE temporal abstraction 是关键，而仅预训练视觉 encoder 在 LIBERO 会负迁移。预训练每任务少于 30 条时表现显著下降，40 条以上较好。
- **局限**：BPE 按频率合并，不感知视觉语义、几何边界或动力学可行性；主要是仿真，尚未验证跨 embodiment 真机精细任务；预训练阶段并非“小样本”。
- **对课题影响｜核心继承 + 风险提示**：它是用户设想最直接的 baseline 之一。可继承“动作 tokenizer + 可变时长技能 + 下游 skill prior”，但需要把 BPE 替换为受状态变化、接触阶段或视觉目标约束的分段；避免把下游 5-shot 与整体数据高效混为一谈。

### 5. Behavior Generation with Latent Actions（VQ-BeT）

- **作者 / 年份 / venue**：Seungjae Lee, Yibin Wang, Haritheja Etukuru, H. Jin Kim, Nur Muhammad Mahi Shafiullah, Lerrel Pinto；ICML 2024 Spotlight（venue 由官方代码仓库说明交叉核验）。[论文](https://arxiv.org/abs/2403.03181)；[官方代码](https://github.com/jayLEE0301/vq_bet_official)
- **skill 类型**：**离散 latent action / action chunk**。Residual VQ 把连续动作 chunk 编码为层级 code，并由 transformer 单次预测 code 与连续 offset；不是语言子任务，也不保证 code 对应人类语义技能。
- **数据与评测**：7 个模拟行为生成任务、nuScenes，以及 Hello Robot Stretch 真机的 5 单阶段、3 双阶段、4 个三至四子任务长时任务。
- **关键结果**：条件任务 6/7 达到 SOTA，无条件任务 5/7 最优；相对扩散推理快约 5 倍（真机约 25 倍）。真机双阶段总成功 19/30，对 Diffusion Policy 11/30；四个长时任务末阶段成功率至少约为 Diffusion Policy 的 3 倍。
- **局限**：优化核心是多模态行为建模，并没有 few-shot 迁移；code 可解释性/跨任务复用性不强；真机长时每任务仅 10 次，置信区间有限。
- **对课题影响｜关键对照 + 评测借鉴**：作为强 latent-action tokenizer 基线；可借鉴分层 residual quantization 和单次推理。论文应额外证明“共通技能”而不只是更强的动作分布拟合。

### 6. QueST: Self-Supervised Skill Abstractions for Learning Continuous Control

- **作者 / 年份 / venue**：Atharva Mete, Haotian Xue, Albert Wilcox, Yongxin Chen, Animesh Garg；NeurIPS 2024（Advances in Neural Information Processing Systems 37；论文 arXiv:2407.15840）。[论文](https://arxiv.org/abs/2407.15840)；[项目页](https://quest-model.github.io/)；[官方代码](https://github.com/pairlab/QueST)
- **skill 类型**：明确的**离散 latent skill 序列**。Stage I 用因果 encoder/decoder 将动作序列压成多个 FSQ code；Stage II 用 GPT 风格 prior 根据视觉和任务逐 token 预测，再解码动作并 receding-horizon 重规划。
- **数据与评测**：LIBERO-90（90 任务、每任务 50 demonstrations）、LIBERO-LONG（10 长时任务）、MetaWorld ML45；在 8 个未见 LIBERO-LONG 和 5 个未见 MetaWorld 任务做 5-shot。
- **关键结果**：LIBERO-90 平均成功 88.6%，VQ-BeT 81.3%，Diffusion Policy 75.4%；LIBERO-LONG 5-shot 为 68.8%，比次优约高 14 个绝对百分点；冻结 skill decoder 只下降约 1.5%。因果结构消融显示它对困难 few-shot 尤其重要。
- **局限**：完全是仿真；held-out tasks 与预训练任务结构相似，作者明确承认更异质新任务可能超出 learned skill space；未利用触觉/力觉/三维几何。
- **对课题影响｜核心继承 + 关键对照**：这是“共通 latent skill 提升低样本成功率”的最强直接依据和首要 baseline。适合继承 two-stage tokenizer/prior，但把 Stage I 扩展为接触感知、多模态且允许可变时长；把目标从相似任务迁移提高到精细接触任务的对象/几何变化迁移。

### 7. Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos

- **作者 / 年份 / venue**：Yi Chen, Yuying Ge, Weiliang Tang, Yizhuo Li, Yixiao Ge, Mingyu Ding, Ying Shan, Xihui Liu；arXiv 2024，ICCV 2025 Oral。[论文](https://arxiv.org/abs/2412.04445)；[项目页](https://chenyi99.github.io/moto/)；[官方代码](https://github.com/TencentARC/Moto)
- **skill 类型**：从相邻视频帧变化提取的**latent motion token**，与机器人 action latent 不完全相同。Moto-GPT 自回归预训练 motion token，下游联合预测 motion token 与真实 action。
- **数据与评测**：109k OXE trajectory videos 预训练；SIMPLER 用 73k RT-1 action 轨迹微调；CALVIN 18k action 轨迹；真机 FANUC 三任务各 30 demonstration；另加 105k Something-Something V2 人类视频。
- **关键结果**：CALVIN 五步平均链长约 3.10，优于去 motion token 的 2.14；真机平均成功从 23.33% 提升到约 60%，视觉干扰/新物体下分别提升约 20/30 个百分点。数据效率实验中仅用 1% action-labeled data 达 52.5%，从头训练为 0%。
- **局限**：大规模视频预训练仍是数据密集；motion token 捕捉视觉变化，不必然对应可执行或接触稳定的技能；真实任务仅 3 个、每任务 10 次测试；表格排版和不同基线数据配方需谨慎横比。
- **对课题影响｜核心继承 + 风险提示**：为“用无动作视频预训练 latent skill，再少量真机对齐”提供最强证据。可将其作为 Stage 0 motion prior，但需要显式 action feasibility / contact consistency，使 motion latent 不停留在视觉层。

### 8. Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models

- **作者 / 年份 / venue**：Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong 等；ICML 2025。[论文](https://arxiv.org/abs/2502.19417)；[项目页](https://www.pi.website/research/hirobot)
- **skill 类型**：**人类标注的原子语言子任务**，不是 latent skill。高层 VLM 结合观察、复杂提示和实时反馈选下一条短语言 skill，低层 flow-matching VLA 执行动作。
- **数据与评测**：完整 demonstrations 切成通常 5-100 actions 的短 skill 并人工标注；另用合成复杂对话数据训练高层。单臂、双臂和移动双臂上评测清桌、做三明治、购物。
- **关键结果**：相对 GPT-4o，高层 instruction accuracy 平均高 40%+；相同数据下层级结构优于 flat VLA。专家提供高层指令时低层几乎无误，表明主要瓶颈是 reasoning，而非该任务集中的 actuation。
- **局限**：原子 skill segmentation/标签昂贵；高层依赖 prompt engineering 合成数据；高低层独立训练，对彼此能力未知；指标偏 instruction accuracy/task progress，不能证明低样本精细 skill 学习。
- **对课题影响｜关键对照 + 风险提示**：语言层适合复杂任务分解、用户反馈和可解释性，但不应成为精细控制唯一瓶颈。可作为高层模块和“语言 skill only”对照。

### 9. π0.5: a Vision-Language-Action Model with Open-World Generalization

- **作者 / 年份 / 载体**：Physical Intelligence / Kevin Black, Noah Brown, James Darpinian 等；2025；arXiv:2504.16054。[论文](https://arxiv.org/abs/2504.16054)；[官方代码（openpi）](https://github.com/Physical-Intelligence/openpi)
- **skill 类型**：统一模型内的**语言子任务预测 + flow-matching 动作专家**；不是显式 latent skill 库。高层输出诸如 “pick up the cup”，再作为低层上下文。
- **数据与评测**：约 400 小时移动操作数据，来自 104 个地点；还混合多环境静态臂、跨 embodiment、web VQA/定位、高层语言和 verbal-instruction data。全在训练未见环境评测，真家庭长时任务持续约 2-5 分钟。
- **关键结果**：训练场景从 3 增至 104 时，新环境任务与语言遵循稳定提升；104 场景模型接近直接见过测试家的 control。显式高层+低层最好；去 verbal instruction、web data 或用零样本 GPT-4 高层明显下降。
- **局限**：增益同时来自数据规模、配方、预训练和层级推理，无法归因于 skill；没有 few-shot 新精细任务；数据和算力规模不适合作为硕士课题直接复现。
- **对课题影响｜评测借鉴 + 外围背景**：说明高层语义分解对于跨场景长时任务有效，但研究应在可控规模上做因果消融，而不是追逐大模型规模。可以 openpi checkpoint 为低层初始化，在其上研究 latent skill adapter。

### 10. SkillVLA: Tackling Combinatorial Diversity in Dual-Arm Manipulation via Skill Reuse

- **作者 / 年份 / 载体**：Xuanran Zhai, Zekai Huang, Longyan Wu, Qianyou Zhao, Qiaojun Yu, Jieji Ren, Ce Hao, Harold Soh；2026；arXiv:2603.03836（截至核验为预印本）。[论文](https://arxiv.org/abs/2603.03836)
- **skill 类型**：以语言/视觉高层识别的**单臂语义技能和协作模式**为主，底层按臂分离 action expert；不是 QueST 式自监督 codebook。离散 gate 决定独立或跨臂交互。
- **数据与评测**：真机 20 个任务、3 个强协作任务、2 个长时任务；训练只出现单臂技能，测试未见左右组合；继续学习用少量双臂 demonstrations。
- **关键结果**：9 个未见左右组合中 π0.5 和 π0-FAST 均 0%，SkillVLA 平均 51%；已见单臂技能保持 78%。协作任务与 π0.5 相当；长时任务完成时间降约 21%。LIFT BAG 可 20% 零样本，仅 5 条微调 demonstration 即接近峰值，baseline 30 条仍未收敛。
- **局限**：2026 新预印本，独立复现和更多机器人验证尚缺；技能结构强依赖双臂可分解先验；高层 VLM 带来延迟/成本；作者也提出未来需超越自然语言的更丰富技能表征。
- **对课题影响｜核心继承 + 评测借鉴**：提供“结构化技能复用能降低 demonstration burden”的最新真机证据。核心可迁移思想不是双臂本身，而是**按可组合因子解耦 action expert，并仅在必要时通信**。精细任务可按“接近/接触/约束运动”或“手臂/手指”因子分解。

### 11. Grounding Hierarchical VLA Models Through Explicit Language-Action Alignment（GPLA）

- **作者 / 年份 / 载体**：Theodor Wulff, Federico Tavella, Rahul Singh Maharjan, Manith Adikari, Angelo Cangelosi；2026；arXiv:2604.05614 / IEEE copyright preprint，正式 venue 尚待核实。[论文](https://arxiv.org/abs/2604.05614)
- **skill 类型**：**语言子任务**。另训练 action-conditioned grounding model，把视觉-动作轨迹与语言嵌入对齐，再用偏好优化选择更 grounded 的语言-动作候选。
- **数据与评测**：LanguageTable；高层 Gemma 3 生成低层语言，SmolVLA 生成 8-step trajectory；单 A100。指标含 BLEU/ROUGE/METEOR/BERTScore 和动作 MSE/MAE/cosine，而非在线成功率。
- **关键结果**：无需新增低层真值标注的 GPLA 在动作误差上接近监督模型；action-conditioned 对比表征比 CLIP/SigLIP 更明显地混合语言与视觉-动作 embedding。
- **局限**：没有真机和在线任务成功率；监督模型总体仍最优；数据小导致对比学习受限；语言指标与物理正确性不等价。
- **对课题影响｜风险提示 + 评测借鉴**：它指出层级 VLA 的核心失败模式：语言中间层可能“说得通但做不对”。如果采用语言 skill，应加入 language-latent/action alignment loss 和 groundedness 指标，但不能以文本重合指标替代成功率、阶段成功率和接触质量。

## 二、横向比较：这里的 skill 到底是什么

| 表征路线 | 代表工作 | 优点 | 对低样本精细任务的直接证据 | 主要缺口 |
|---|---|---|---|---|
| 语言子任务 | RT-H、Hi Robot、π0.5、GPLA | 可解释、可人工纠正、适合复杂任务分解与组合 | RT-H 总体 +15%，但没有 few-shot；精细任务中动作簇偶有优势 | 丢失几何/力/速度细节；标注与 grounding 成本 |
| 视觉子目标 | SuSIE | 保留物体、末端姿态和局部几何；低层目标更具体 | Oracle 对照显示真机 0.78->0.88、CALVIN 0.66->0.95 | 可达性未知、生成幻觉、难表示力和隐状态 |
| 动作 latent skill | PRISE、VQ-BeT、QueST | 自监督、压缩时间、跨任务复用、直接连接控制 | PRISE 5-shot +9.2pp；QueST 5-shot 68.8%、+14pp | 仿真为主、语义/边界不稳定、跨 embodiment 困难 |
| 视频 latent motion | Moto | 无需 action label，可用人类/机器人视频扩规模 | 1% action data 52.5% vs 0%；真机约 23.3%->60% | 视觉运动不保证可执行和接触稳定 |
| 结构化可组合 expert | SkillVLA | 抑制技能纠缠，可零样本重组，适合多部件控制 | 未见双臂组合 0->51%；5 demos 接近峰值 | 依赖正确的可分解结构和高层模式判断 |

## 三、对用户课题的高价值判断

1. **最有论文价值的定义不是“用 skill 提高成功率”，而是“学习与任务语义解耦、与接触动力学相关、可组合且可少样本适配的 latent skill”**。语言子任务只是高层接口，不能作为隐特征创新本身。
2. **层级提高精细度的关键是缩短低层目标距离和降低动作多模态**。SuSIE 的 oracle 对照最有说服力：最终目标相同，中间目标仍显著提高控制精度。因此应让 latent skill 预测短时、可达的视觉/几何变化，而非只压缩 action。
3. **低样本不是少数据预训练**。PRISE/QueST/Moto 都依赖大量跨任务或视频预训练，少的是目标任务 action demonstrations。论文必须分别报告预训练数据量和下游 1/5/10/20-shot，避免夸大“高效样本学习”。
4. **精细任务的 skill 边界应由接触事件而非固定 chunk 决定**。推荐边界信号：夹爪状态变化、末端速度/方向突变、接触/力阈值、object-centric state change、视觉关键点拓扑改变。固定长度 token 是必要对照。
5. **共享和专用必须共存**。完全共享会造成 negative transfer / skill entanglement；完全专用失去样本效率。SkillVLA 的离散 gating 提供可实现原则：共享 latent expert 处理共通段，task-specific residual 或接触 expert 处理精细差异。
6. **最强可行路线是三层而非单一 skill token**：语言子任务解决“做什么”，视觉/几何子目标解决“到哪里/形成什么关系”，latent action skill 解决“怎么稳定做到”。三层间需要 feasibility/grounding 对齐并闭环重规划。

## 四、建议形成的硕士课题

### 暂定题目

**Grounded Contact-Aware Latent Skill VLA for Few-Shot Fine-Grained Manipulation**  
中文：**面向低样本精细操作的可落地接触感知隐技能视觉-语言-动作模型**

### 核心研究问题

- RQ1：跨任务预训练得到的 latent skill，相比 raw action chunk、语言 skill 和纯动作 VQ，能否在 1/5/10-shot 新精细任务上提高成功率？
- RQ2：以接触/物体状态变化学习可变时长边界，是否优于固定长度 chunk 和 PRISE-BPE？
- RQ3：视觉/几何子目标与 action latent 的一致性约束，是否减少“规划正确、执行不精确”及 subgoal 不可达？
- RQ4：共享 skill + task residual / gated expert 是否减少跨任务负迁移和 skill entanglement？

### 建议架构

1. **High-level semantic planner**：复用轻量 VLM/VLA，根据图像和任务生成短语言子任务；不把它声称为 latent skill。
2. **Contact-aware skill tokenizer**：输入短轨迹的 RGB（最好加 wrist RGB）、proprioception、gripper state；有条件时加 force/torque。用 FSQ/RVQ 学离散 skill token，边界由接触或 object-state-change proposal 产生。
3. **Grounded subgoal head**：每个 token 同时预测局部目标（关键点/物体相对位姿/未来图像 latent），通过 inverse dynamics、future-state consistency 和 reachability loss 约束可执行性。
4. **Skill prior**：VLA 根据当前观测、语言子任务和历史 token 自回归选择 latent skill；高频低层 decoder 输出 action chunk，receding horizon 执行。
5. **Adaptive specialization**：共享 decoder + 小型 task/contact residual；离散 gate 在自由空间、接触、约束运动之间选择或控制专家通信。

### 训练策略

- Stage A：在多任务轨迹上自监督学习 tokenizer/decoder；action reconstruction 之外加入 future observation、inverse dynamics、temporal consistency、code usage regularization。
- Stage B：冻结或半冻结 tokenizer，训练 vision-language skill prior；用自动切分后的小段字幕或 VLM 伪标签对齐语言，但 latent 不由语言标签定义。
- Stage C：新任务只用 1/5/10/20 demonstrations，比较仅训 prior、prior+residual、全量微调；可加入失败片段或 DAgger 式少量干预。

### 最小可行评测

- **仿真**：LIBERO-90 预训练，LIBERO-LONG 5-shot 作为与 PRISE/QueST 对齐的标准结果；再构造 precision perturbations（目标容差、物体尺寸、摩擦、视觉干扰）。
- **真机精细任务**：插拔/对孔、瓶盖旋拧、拉链或线缆插接、薄片抓取/精确放置中选 3-4 项；保证共通阶段（接近、预抓、接触、约束移动、释放）可跨任务复用。
- **数据曲线**：0/1/5/10/20-shot；每设定至少 3 seeds，真机每任务每条件建议 20+ trials，Wilson 区间。
- **指标**：最终成功率、阶段成功率、末端/物体位姿误差、接触峰值力/滑移、完成时间、重规划次数、每成功任务 demonstrations 数；另报告 token 使用熵、跨任务 token 重合、线性探针和检索纯度。
- **关键消融**：flat VLA；固定 action chunk（ACT）；纯 RVQ/VQ-BeT；PRISE；QueST；语言 skill only（RT-H 风格）；无 subgoal grounding；固定边界；无 residual/gate；oracle segmentation；oracle subgoal。

### 可以成立的创新点

- **接触感知可变时长 latent skill tokenizer**，而不是再做一种普通 action quantizer。
- **视觉/几何子目标与 action skill 的可达性对齐**，补足 SuSIE 与语言层级高低层解耦的问题。
- **共享 latent skill + 少量专用 residual 的低样本适配机制**，直接针对 negative transfer / skill entanglement。
- **对“skill 是否共通”的可证伪评测协议**：冻结 decoder、跨任务 token 复用、组合未见技能和严格 shot curve，而非只展示 t-SNE。

## 五、风险与范围控制

- 不建议从头训练“大模型”；硕士周期内应复用 OpenVLA/openpi/SmolVLA 或现有 policy backbone，把贡献集中在 skill tokenizer、对齐和适配。
- 不建议一开始同时做双臂、触觉、互联网视频和真实机器人。优先完成单臂 RGB+proprio 的仿真闭环，再选择一种额外模态和 3 个真机任务。
- 若真机没有力传感器，可用 gripper current、末端速度突变、视觉物体位移作为弱接触信号，并设置 oracle simulator contact 作为上界。
- 2026 文献（SkillVLA、GPLA）较新，结论需视为前沿预印本证据；正式论文中应在投稿前再次核实版本、代码和后续独立复现。

## 六、证据优先级

- **核心继承**：QueST、PRISE、SuSIE、Moto、SkillVLA。
- **关键对照**：RT-H、VQ-BeT、Hi Robot、ACT / flat VLA。
- **评测借鉴**：LIBERO 5-shot 协议、SuSIE oracle-goal 对照、SkillVLA 未见组合与 demonstration 曲线、长时阶段成功率。
- **风险提示**：GPLA（语言-动作不一致）、PRISE（预训练数据阈值）、QueST（结构相似 held-out）、SuSIE（不可达/幻觉子目标）。
- **外围背景**：π0.5（规模化层级 VLA 数据配方）、PlayFusion（非专家 play 与离散瓶颈）。

## 七、核验说明

- 题名、作者、年份和实验数字优先来自论文 PDF；venue 同时参考论文页、arXiv comment、PMLR/项目页或官方代码说明。
- “低样本有效”只在论文存在明确 shot 数和对照时标注。RT-H、Hi Robot、π0.5、SuSIE 的优势不等价于 few-shot 证据。
- 2026 年条目以 2026-08-13 可访问版本为准；未能确认正式 venue 时明确标为预印本，未作推断。
