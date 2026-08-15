# 动作 Tokenizer、Action Chunking 与跨本体表示文献调研

> 范围：VLA 动作表示、动作分块、离散 tokenizer、跨本体控制，以及它们对精细操作和样本效率的影响。  
> 证据标准：优先论文原文、作者项目页和官方代码；预印本不误标为已接收论文。  
> 最后更新：2026-08-13（持续增量更新）

## 名称消歧

- **FAST**：Pertsch 等，*FAST: Efficient Action Tokenization for Vision-Language-Action Models*，arXiv:2501.09747（2025）。
- **CrossFormer**：模型名出现在 Doshi 等的 *Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation*，arXiv:2408.11812（2024）；不要与同名视觉 Transformer 混淆。
- **Learning to Act Anywhere**：正式题名为 *UniVLA: Learning to Act Anywhere with Task-centric Latent Actions*，Bu 等，RSS 2025，arXiv:2505.06111。
- **TACO**：Shiarlis 等的 *TACO: Learning Task Decomposition via Temporal Alignment for Control*，ICML 2018；它研究弱监督的任务分解和对齐，并非近期通用动作 tokenizer。
- **UniAct（机械臂语境）**：Zheng 等的 *Universal Actions for Enhanced Embodied Foundation Models*，CVPR 2025，arXiv:2501.10105；模型名 UniAct，学习跨机器人 universal action codebook。
- **UniAct（人形语境）**：Jiang 等的 *UniAct: Unified Motion Generation and Action Streaming for Humanoid Robots*，arXiv:2512.24321（2025）；是人形全身运动的共享 FSQ codebook。两者同名但不是同一工作。

## 已核验核心工作

### FAST：频域动作序列 token，而非语义 skill

**引用。** Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, Sergey Levine. *FAST: Efficient Action Tokenization for Vision-Language-Action Models*. 2025，arXiv:2501.09747。官方项目页：https://www.pi.website/research/fast 。

**方法。** 对归一化动作 chunk 沿时间维作 DCT，把小的频域系数量化为整数，再通过 byte-pair encoding 压缩为可变长离散 token 序列。FAST+ 在约 100 万条真实机器人轨迹上训练为通用 tokenizer；其目标是降低自回归 VLA 的序列长度与训练难度，不显式学习 task/outcome/contact 语义。

**论文主张。** FAST 与 π0 结合可扩展到约 10,000 小时数据，训练时间最高减少 5 倍，同时达到 diffusion VLA 的性能水平；标准逐维逐时刻离散化在高频灵巧任务上可能完全失败。

**代码。** OpenPI 官方仓库提供 FAST/π0-FAST 模型与 tokenizer 接口（`src/openpi/models/pi0_fast.py`、`tokenizer.py`；https://github.com/Physical-Intelligence/openpi ，Apache-2.0）。FAST+ tokenizer 权重另由官方 Hugging Face 资源分发。

**局限与 CASK 影响。** FAST 是 CASK 必须纳入的强动作压缩对照，但压缩 token 不等于可复用 skill。CASK 可以把 FAST/连续 flow matching 作为 π0 低层执行器，在上层增加事件驱动边界、跨任务 outcome/contact 一致性和离散 skill；需实验证明语义层相对“仅换 tokenizer”的额外收益。若 CASK 声称“可变长动作 token”或“通用动作离散化”为创新，会与 FAST 直接冲突。

### ACT：action chunking 对精细双臂操作的强基线

**引用。** Tony Z. Zhao, Vikas Kumar, Sergey Levine, Chelsea Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*. RSS 2023，DOI:10.15607/RSS.2023.XIX.016，arXiv:2304.13705。项目/代码：https://tonyzhaozh.github.io/aloha/ ，https://github.com/tonyzhaozh/act 。

**方法。** Action Chunking with Transformers（ACT）以 CVAE Transformer 一次预测固定长度动作块；时间集成将相邻时刻预测的重叠 chunk 加权平均，缓解误差累积并平滑执行。潜变量主要吸收示范风格，不是跨任务离散 skill。

**论文结果。** 用约 10 分钟、约 50 条示范即可学习六个精细双臂任务；真机成功率总体达到 80%–90% 量级，并显著优于逐步行为克隆。具体逐任务数字将在实验表中按原文补齐。

**局限与 CASK 影响。** 固定 chunk 边界可能跨越接触建立/释放，时间集成也可能平滑掉需要突变的夹爪或接触动作。它是检验 CASK “事件边界优于固定 H” 的首要低数据基线：固定 H ACT、不同 H 扫描、带/不带 temporal ensemble、事件切分但无 skill code 都应分别比较。

### Diffusion Policy：连续多峰 action chunk 的性能上界型基线

**引用。** Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. RSS 2023；IJRR 2024，DOI:10.1177/02783649241273668，arXiv:2303.04137。项目/代码：https://diffusion-policy.cs.columbia.edu/ ，https://github.com/real-stanford/diffusion_policy 。

**方法。** 以条件去噪扩散建模连续动作序列，使用 receding-horizon action prediction/action execution；能表达多峰动作分布，并通过 chunk 保持时间一致性。

**论文结果。** 在 4 个机器人操作基准、12 个任务上平均比既有方法提升 46.9%；官方代码和配置公开。

**局限与 CASK 影响。** 推理需要多步去噪，离散语义与边界不可解释，且固定预测窗不能直接表达事件持续时间。CASK 应将其作为连续 chunk 上限和“离散 skill 是否牺牲精度”的关键对照；连续 residual 分支应直接对比 diffusion/flow action head，而不能只对比逐步 MSE BC。

### RT-H：语言动作层级，但不是自发现 latent skill

**引用。** Suneel Belkhale, Tianli Ding, Ted Xiao, Pierre Sermanet, Quoc Vuong, Jonathan Tompson, Yevgen Chebotar, Debidatta Dwibedi, Dorsa Sadigh. *RT-H: Action Hierarchies Using Language*. 2024，arXiv:2403.01823。

**方法。** 将机器人动作先表示为称作“语言运动（language motion）”的中间自然语言短语，再由低层策略生成动作；层级接口允许人类在执行时纠正中间动作指令。

**论文结果。** 在超过 2,500 小时的多任务真机数据上，RT-H 相对 RT-2-X 绝对成功率提升 15%；使用约 27% 的异构数据仍可达到 RT-2-X 的性能，并通过语言运动干预显著改善执行。

**代码。** 截至本次核验，论文未给出官方训练代码仓库；应把它视为方法与实验对照，而不是可直接复现基线。

**局限与 CASK 影响。** 中间标签来自语言/自动标注，粒度和边界未由 contact/outcome 约束，语言也难表达高频精细残差。它是 CASK “离散 skill 是否比语言 motion 更紧凑、更可复用”的关键对照；CASK 可保留语言可解释映射，但底层应由 π0 的连续 action expert 执行。

### TACO：带任务草图的弱监督边界学习先例

**引用。** Kyriacos Shiarlis, Markus Wulfmeier, Sasha Salter, Shimon Whiteson, Ingmar Posner. *TACO: Learning Task Decomposition via Temporal Alignment for Control*. ICML 2018（PMLR 80），arXiv:1803.01840。论文页：https://proceedings.mlr.press/v80/shiarlis18a.html 。

**方法。** 给定每条示范的高层任务 sketch（子任务符号序列），通过类似 CTC 的动态规划联合对齐未知时间边界并训练子策略；不需要逐帧子任务标签。

**局限与 CASK 影响。** TACO 已覆盖“从弱监督序列学习分段边界”，因此 CASK 不能把该宽泛命题作为创新。其依赖预先给定的 skill 顺序和离散子任务词典，且时代较早、任务规模小；CASK 的差异应落在无 sketch 的事件驱动边界、跨任务 outcome/contact 一致性，以及离散 skill 加连续 residual 的 VLA 实现。

### CrossFormer：跨本体共享骨干，不强制共享动作坐标

**引用。** Ria Doshi, Homer Walke, Oier Mees, Sudeep Dasari, Sergey Levine. *Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation*. 2024，arXiv:2408.11812。项目/代码：https://crossformer-model.github.io/ （论文项目页；代码状态以项目页为准）。

**方法。** 将不同数量的图像、本体状态和动作 readout token 序列化，使用共享 decoder-only Transformer；不同 embodiment class 使用独立连续 L1 action head，动作维度和 chunk 长度均可不同。它没有学习一个共同动作坐标，而是通过共享视觉骨干和固定 token 槽位联合训练异构头。

**数据与结果。** 约 90 万条轨迹、20 种 embodiment，涵盖单臂、双臂、地面导航、四足和无人机。真机评测平均成功率为 73%，同架构单本体训练为 67%，各领域已有专用方法平均为 51%。模型约 130M 参数；论文明确指出尚未观察到显著的跨本体正迁移，并认为大模型推理速度会限制高频精细控制。

**CASK 影响。** CrossFormer 是“共享 policy backbone + 本体专用低层头”的直接基线，也支持 CASK 让 π0 承担本体专用解码、skill 只在高层共享。CASK 必须报告 per-embodiment 正迁移/负迁移，而不能只报告混合平均值；其创新应是 outcome/contact 对齐后的共享离散 skill，不是“一个网络控制多种机器人”。

### UniVLA：与 CASK 最接近的 task-centric 离散 latent action 竞争工作

**引用。** Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, Hongyang Li. *UniVLA: Learning to Act Anywhere with Task-centric Latent Actions*. RSS 2025，arXiv:2505.06111。代码：https://github.com/OpenDriveLab/UniVLA （公开，Apache-2.0）。

**方法。** 由相邻视频帧的 inverse dynamics 用两阶段 VQ-VAE 提取 task-centric latent action：先吸收任务无关视觉变化，再冻结该 codebook 并学习任务相关 codebook；VLM 自回归预测离散 latent action，轻量动作 decoder 将其解码为目标机器人 action chunk。默认每个 latent action 覆盖约 1 秒，codebook 大小 16；action head 约 12.6M 参数，使用 LoRA 适配。

**数据、算力和结果。** 预训练使用 Open X-Embodiment 子集、GNM 和 Ego4D；论文报告 960 A100-hours，而 OpenVLA 为 21,500 A100-hours。相对 OpenVLA，LIBERO 平均成功率提升 18.5 个百分点，导航提升 29.6 个百分点，真机提升 36.7 个百分点；LIBERO-Goal 仅用 10% 示范达到 86.3%，高于 OpenVLA 全数据的 79.2%。Piper 真机平均成功率 68.9%，LAPA 28.9%，OpenVLA 20.0%；RTX 4090 上以 chunk=12 达到 10 Hz。代码公开。

**局限与创新冲突。** 论文自己承认 latent action 粒度固定、codebook 大小预设，主要验证单臂，双臂/灵巧手需要更细粒度表示。它已基本占据“task-centric 离散 latent action + 跨本体视频预训练 + 少量目标动作解码器”命题，是 CASK 的最关键对照和最大创新冲突。CASK 必须把差异压实为：事件驱动可变边界、显式 outcome/contact 跨任务一致性、离散 skill 之外的连续 residual，以及 π0 闭环低层执行；消融需包含 UniVLA 式固定约 1 秒 latent token。

### UniAct（CVPR 2025）：离散 universal atomic behavior 的直接竞争工作

**引用。** Jinliang Zheng, Jianxiong Li, D. G. Liu, Yinan Zheng, Zhihao Wang, Zhonghong Ou, Yu Liu, Jingjing Liu, Ya-Qin Zhang, Xianyuan Zhan. *Universal Actions for Enhanced Embodied Foundation Models*. CVPR 2025，arXiv:2501.10105。项目页：https://2toinf.github.io/UniAct/ ；官方代码：https://github.com/2toinf/UniAct （公开，仓库未声明 SPDX license）。

**方法。** 用 VLM 从 observation 和 goal 提取跨本体“原子行为”，经 Gumbel-Softmax 选择 256×128 的 VQ universal action codebook；轻量 embodiment-specific decoder 将共享 code 还原为不同控制接口。0.5B 模型在 28 种 embodiment、约 100 万 demonstrations 上训练。

**结果。** 论文称 0.5B UniAct 在多项仿真/真机评测优于 14 倍参数规模的 OpenVLA 7B。新 AIRBOT 上使用 100 条 demonstrations 分别适配相对/绝对 EEF 与 joint 控制，只训练约 4M 参数（总模型 0.8%）；双臂 AIRBOT 的四个任务各收集 250 demos。人工检查 256 个 universal actions，在不同机器人上至少约 40% 呈现完全一致的解码行为。项目页公开；代码开放程度需按仓库进一步确认。

**创新冲突。** 该工作已经明确主张“离散共享 atomic behavior + 本体专用 decoder + 高效新机器人适配”，与 CASK 的离散 skill/π0 解码结构高度重合。CASK 必须把 code 的语义形成机制区别开：不是仅由 goal/action reconstruction 学习，而是由可变事件边界和跨任务 outcome/contact 一致性约束；并需证明连续 residual 在精细接触任务上优于纯 code 解码。

### UniAct：共享离散运动流的实时人形控制

**引用。** Nan Jiang, Zimo He, Wanhe Yu, Lexi Pang, Yunhao Li, Hongjie Li, Jieming Cui, Yuhan Li, Yizhou Wang, Yixin Zhu, Siyuan Huang. *UniAct: Unified Motion Generation and Action Streaming for Humanoid Robots*. 2025，arXiv:2512.24321。项目页：https://jnnan.github.io/uniact/ 。

**方法。** 以 FSQ 对运动和音乐分别编码，将文本、音乐、轨迹和运动 token 拼到 Qwen2.5-3B 的共享词表；因果 decoder 将生成 token 流式解码为连续自由度目标，再交给 BeyondMimic tracker。机器人运动 FSQ 使用 5 个 latent 维度和量化级别 [8,8,8,6,5]，共 15,360 个 code；chunked decoding 在延迟和时序分辨率间折中。

**数据与结果。** UniMoCap/UA-Net 包含约 20 小时文本配对动作，轨迹增强把 20 分钟步行扩展至 10 小时以上；评测超过 1,000 次仿真和 100 小时真机运行。系统新指令响应低于 500 ms；摘要报告对有缺陷/OOD 参考运动的 zero-shot tracking 成功率提高 19%。论文项目页可用，代码是否完整发布需单独复核。

**CASK 影响。** 它证明“离散 motion manifold + 连续 tracker”能提升 OOD 稳定性，与 CASK 的“离散 skill + π0 低层执行”结构相似；但其 token 是全身运动形态，边界由固定下采样/chunk 决定，未建模 manipulation outcome/contact。CASK 不能笼统声称首次以离散上层 token 驱动连续控制器，应突出接触事件和精细物体操作。

### CLAM：连续 latent action 是精细控制中离散化的强反证

**引用。** Anthony Liang, Pavel Czempin, Matthew M. Hong, Yutai Zhou, Jingzhen Wang, Erdem Biyik, Stephen Tu. *CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations*. 2025，arXiv:2505.04999。项目/代码：https://clamrobot.github.io/ 。

**方法。** inverse/forward dynamics 从 observation-only trajectory 学习连续 latent action，并用少量带动作的 play data 联合训练 action decoder，使 latent 容易落地为真实连续动作。与 VQ latent action 相反，CLAM 明确主张精细连续控制不应先离散化。

**结果。** 在 DMControl、MetaWorld 和四个 WidowX 真机任务上评估；约每任务 30 条无动作 expert demonstration，并用较易采集的带动作 play data 对 decoder grounding。论文报告相对最佳 latent-action baseline 的任务成功率提升约 2–3 倍，MetaWorld 图像任务示例达到 76% 并约为最佳对照 3 倍；代码公开。

**CASK 影响。** 这是 CASK “离散 skill”方案必须正面回答的反证：单纯 VQ 很可能损伤接触阶段的精度。连续 residual 不能只是附属模块，需展示它降低 quantization error、提高插入/对齐/旋拧成功率；关键对照为 continuous latent only、discrete only、discrete+residual，并报告动作重建误差与接触峰值误差。

### UHAS 与 Cloak：2026 跨本体转移表明动作对齐未必靠 skill token

**UHAS 引用与方法。** Luis Felipe Casas, Robert Teal, Keval Shah, Abhijit Tadepalli, Wanxin Jin, Yu Xiang. *Cross-Embodiment Robot Manipulation via a Unified Hand Action Space*. 2026，arXiv:2607.03570。以 canonical sphere deformation 表示手部动作，再用 Cascade IK 映射到 Allegro、LEAP、Shadow 和 MANO。代码/数据/视频：https://irvlutd.github.io/UHAS/ 。从 MANO 向新手只微调 500 iterations，而从头训练约 4,500 iterations；支持 zero-shot 和真机方块重定向。

**Cloak 引用与方法。** *Cloak: Zero-Shot Cross-Embodiment Manipulation by Masking the End-Effector from the VLA*. 2026，arXiv:2606.22836。基于 π0.5，在训练时从腕部图像注意力中遮蔽末端执行器，并以双指尖 pose retargeting/IK 对齐动作；只用原始平行夹爪约 70K episodes，零样本部署到新夹爪、新机械臂和五指手。Sharpa 手上 progression rate 达 81.8%；论文明确限制为可由两个指尖表达的动作，不覆盖需要新接触策略的灵巧操作。

**CASK 影响。** 两者构成重要替代解释：跨本体收益可能来自几何 canonicalization、IK 或消除视觉 embodiment shortcut，而非学到共通 skill。CASK 若做跨本体实验，应加入统一末端/指尖空间、masking/retargeting 基线，并选择接触策略真正共享但运动学不同的任务；否则无法把提升归因于 outcome/contact skill。

### Latent Action Diffusion 与 OPFA：跨本体 latent 已进入显式几何对齐阶段

**Latent Action Diffusion。** Erik Bauer, Elvis Nava, Robert K. Katzschmann. *Latent Action Diffusion for Cross-Embodiment Manipulation*. ICRA 2026，arXiv:2506.14608。项目/代码：https://mimicrobotics.github.io/lad/ 。用 retargeted 对齐末端姿态构造正对，InfoNCE 学连续共享 latent action，再用 modality-specific decoder 和 diffusion policy。Faive/Franka 各 100 episodes 的真机 pick-place 中，跨本体联合训练比单本体提高 10% 和 7.5%；Mimic/Franka 各 250 episodes 时 Mimic 提高 13%，但 Franka 出现负迁移。说明共享 latent 可能正迁移，也可能被相机缺失和数据不对称破坏。

**OPFA。** Juncheng Mu, Sizhe Yang, Hojin Bae, Feiyu Jia, Qingwei Ben, Boyi Li, Huazhe Xu, Jiangmiao Pang. *One-Policy-Fits-All: Geometry-Aware Action Latents for Cross-Embodiment Manipulation*. ICRA 2026，arXiv:2603.14522。项目页：https://mujc2021.github.io/opfa/ 。从各末端可达状态点云学习 Geometry-Aware Latent Representation，并用单一 latent-retargeting decoder 覆盖 11 种末端和 14 个任务；跨本体联合训练成功率相对单源最高提高 50% 以上，新 embodiment 仅 8 条 demonstrations 可达到从 72 条 demonstrations 训练的水平。

**CASK 影响。** “共享 latent + 跨本体少样本”本身已经拥挤。CASK 需要在同动作对齐质量下比较 LAD/OPFA；并把结果拆为语义迁移、几何 retargeting 和 observation asymmetry 三部分，特别报告负迁移。

### 2026 Contact 直接竞争：Transferring Contact, Not Just Motion

**引用。** Soofiyan Atar, Yao-Ting Huang, Michael Yip. *Transferring Contact, Not Just Motion: Compliant Grasping Across Dexterous Hands*. 2026，arXiv:2606.15516。项目页：https://transferring-contact-not-just-motion.github.io/ ；截至本次核验未发现公开代码仓库。

**方法。** 将各手原始 effort 经系统辨识标定到物理 joint torque/fingertip force 和负载位置描述符，与共享 MANO pose latent 组成跨本体 force-position interface；mask-aware flow-matching policy 在视觉遮挡时依赖 calibrated contact，并由混合 force-position controller 执行。每只手收集 50 条 teleoperation demonstrations。

**结果。** 三种不同灵巧手、10 个刚性/柔性物体上，平均 reach-and-lift 成功率由 0.44 提升至 0.71；在相同 latent/action interface 下相对 Latent Action Diffusion 从 0.19 提升至 0.76；还能零样本迁移到未见的 15-DoF 手配置。限制是每只新手需要逐指系统辨识，控制增益仍有本体/任务特定成分。

**关键创新冲突。** 这篇工作已经直接提出跨本体 contact 表示，因此 CASK 不能声称首次通过 contact consistency 实现跨本体迁移。CASK 可保留的区别是从示范中学习**事件边界与跨任务 outcome/contact 离散语义**，而非把力标定为统一物理坐标；必须把该方法或其 calibrated-force baseline 纳入灵巧手实验，并解释 π0 是否能接收/利用力触觉状态。

## 对 CASK 的实验设计结论

| 要证明的主张 | 必须对照 | 建议指标 |
|---|---|---|
| 事件边界优于固定 chunk | ACT、Diffusion Policy、UniVLA 固定约 1 秒 token；事件切分但无 skill | 成功率、边界 F1/容差、接触建立延迟、重规划次数 |
| 离散 skill 有跨任务复用 | RT-H language motion、UniVLA latent action、continuous CLAM | few-shot 曲线、skill mutual information、跨任务检索 purity、负迁移 |
| 连续 residual 保住精度 | discrete only、continuous only、discrete+residual | action reconstruction、末端误差、峰值力/滑移、精细任务成功率 |
| 跨本体来自语义而非几何捷径 | CrossFormer、UniAct、LAD/OPFA、UHAS、Cloak/IK | seen/unseen embodiment、正迁移量、同 skill 跨本体 outcome/contact 一致性 |
| contact consistency 超过物理标定 | calibrated force-position interface、无 contact、原始传感器拼接 | 力峰值、滑移/压坏率、跨手 calibration 成本、接触阶段迁移 |
| π0 低层执行的价值 | 原生 π0/π0.5、冻结 π0 + skill、全量微调 | 数据量-成功率、推理延迟、灾难性遗忘、OOD 恢复率 |

## 总体判断

动作领域已经分别有高效 tokenization（FAST）、固定 action chunk（ACT/DP）、语言层级（RT-H）、跨本体多头共享（CrossFormer）、universal discrete atomic action（UniAct）、task-centric latent action（UniVLA）、连续 latent action（CLAM/LAD）和跨本体 calibrated contact（Atar 等，2026）。因此 CASK 可成立的窄而强的创新核心不是“skill token”本身，而是 **由接触/outcome 事件定义的可变时长 skill 边界，在跨任务上用 outcome/contact 一致性约束离散语义，并以连续 residual 和 π0 action expert 保留精细控制**。最危险的重合是 UniAct/UniVLA，最强的离散化反证是 CLAM，最直接的 contact 冲突是 *Transferring Contact, Not Just Motion*，最容易遗漏的替代解释是 UHAS/Cloak/OPFA 的几何与视觉对齐。

## 尚待持续追踪

- 2025–2026 论文仍可能更新版本；UniAct 和 2026 预印本的代码完整性、复现脚本与正式 venue 需定期复核。
- 需要进一步补充直接使用力/触觉事件学习 action chunk/skill boundary 的工作；这决定 CASK 的 contact consistency 是否已有更近竞争者。
