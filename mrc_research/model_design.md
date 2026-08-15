# CASK 模型详细设计与决策解析

> 规格版本：v0.2，2026-08-14  
> 上位依据：`research_report.md` v0.2  
> 状态：实现级预规格；数据相关阈值与资源参数需在 M0 后冻结

## 0. 文档目的

本文把研究报告中的 CASK（Contact-Aware Shared Skills）从概念结构细化为可训练、可部署、可消融的模型规格，并记录每个关键选择的依据、替代方案和证伪条件。

本设计只主张：**同一本体、多任务、精细操作中的共享技能和 held-out 新任务少样本迁移**。跨本体、触觉、多源互联网视频和端到端语言规划均作为扩展，不进入最小核心模型。

## 1. 从报告导出的设计要求

| 编号 | 报告结论或风险 | 模型必须满足的要求 | 验证证据 |
|---|---|---|---|
| R1 | 动作压缩不等于技能 | 离散码必须预测对象后果/接触，并能跨任务交换 | outcome/contact probe、skill swap |
| R2 | 固定 chunk 不足以表达事件 | 技能持续时间由事件边界决定 | 固定窗对照、boundary F1、成功率 |
| R3 | 纯离散会损害精细度 | 使用离散共享 mode 与连续实例 residual | discrete/continuous/`z+r` 消融 |
| R4 | task-ID 泄漏会伪造共享性 | tokenizer 不输入语言/task ID/绝对位置，并加入跨任务配对与泄漏探针 | task probe、held-out composition |
| R5 | 层级规划可能生成不可达目标 | 子目标必须是短时、对象中心且带可达性约束 | subgoal error、低层到达率、oracle 对照 |
| R6 | 训练时能编码完整轨迹不代表部署时能预测技能 | 明确区分轨迹后验 `q(z,r\|segment)` 与在线先验 `p(z,r\|context)` | prior-only rollout，禁止测试时使用 posterior |
| R7 | canonical padding 可能污染控制 | canonical 表示只用于技能分支；动作损失在本体原生有效维计算 | mask 检查、原生动作基线 |
| R8 | 接触信息可能只是代理信号 | 所有 contact 标签带来源与置信度 mask | 真值/代理分层报告、无接触消融 |
| R9 | 共享会引发负迁移 | 保留实例 residual，并在真正新技能任务上报告负迁移 | shared vs direct FT、负迁移率 |
| R10 | 主张必须可证伪 | 预先设置停止门槛，不以可视化替代任务证据 | 第 12 节验收门槛 |

## 2. 总体模型

### 2.1 三层职责

```text
图像 + 本体状态 + 可选力觉 + 任务语言
                  │
                  ▼
       L1 语义子任务接口（冻结/外部提供）
              “现在做什么”
                  │
                  ▼
       L2 对象中心可达子目标 g_k
       “到哪里、形成什么对象关系”
                  │
                  ▼
       L3 CASK 技能层
       事件边界 → 离散 mode z_k + 连续 residual r_k
                  │
                  ▼
       π0 action expert + skill adapter
                  │
                  ▼
       短 action chunk；执行前缀后重新观测
```

主实验只训练 L2、L3 和低层适配器。L1 的主协议是把数据集原始任务指令原样传入，不额外使用人工子任务标签；冻结 planner 作为扩展，人工子任务作为 oracle 上界，三者不混合汇报。这样可以把收益归因到技能机制，而不是语言模型规模或额外监督。

### 2.2 训练与部署必须分开

训练时存在两个编码路径：

- **轨迹后验** `q_φ(z_k,r_k | x_{t_k:t_{k+1}}, a_{t_k:t_{k+1}})`：看完整动作段，提供稳定训练目标；
- **在线先验** `p_ψ(z_k,r_k | c_{t_k}, l, g_k, h_{<k})`：只看当前及历史信息，是部署时唯一允许使用的路径。

其中 `c_t` 是当前视觉/本体上下文，`l` 是任务语言，`g_k` 是短时子目标，`h_{<k}` 是历史技能。训练用 posterior-prior 对齐把后验蒸馏到先验。所有主结果必须使用 **prior-only rollout**；若测试时从真实未来片段编码 `z/r`，结果仅能标为 oracle reconstruction，不能计入策略性能。

这是现有报告最需要补全的闭环：没有先验/后验分离，tokenizer 能重建轨迹也不代表机器人能在线选择技能。

## 3. 输入、时间尺度与对象中心表示

### 3.1 输入字段

每个时刻保留：

| 符号 | 内容 | 必需性 |
|---|---|---|
| `I_t^ext` | 外部多视角 RGB | 必需 |
| `I_t^wrist` | 腕部 RGB | 强烈建议；接触遮挡时更有价值 |
| `p_t` | 关节、末端、夹爪、本体速度 | 必需 |
| `a_t` | 本体原生连续动作 | 训练必需 |
| `f_t` | 腕力矩/触觉/电流 | 可选，必须带传感器 mask |
| `m_t` | canonical state/action 有效维 mask | 必需 |
| `l` | 任务语言 | 仅在线先验/规划层使用 |
| `id_task` | 任务编号 | 只用于采样、对抗头和评测，禁止输入 tokenizer |

张量接口使用符号维度，避免在未知机器人配置下伪造固定值：图像为 `[B,V,H,W,3]`，技能序列为 `[B,L,*]`，本体原生 action chunk 为 `[B,H_a,D_a]`。三类 mask 必须分开：

- `M_time [B,L]`：时间/段 padding mask，只用于 temporal attention 和序列 loss；
- `M_feature [B,L,D_c]`：canonical feature 有效维，在输入投影前逐维相乘并作为额外特征输入；
- `M_sensor [B,L,D_s]`：传感器存在性/置信度，控制 force/contact 等 loss，不得把缺失传感器当零测量。

所有归一化统计量只在训练 split 拟合。位置使用米、角度使用弧度、力使用牛顿、力矩使用牛顿米；数据适配器必须保存源单位和时间戳。`T_A^B` 在本文统一表示“坐标系 B 在坐标系 A 中的位姿”。

### 3.2 双时间尺度

- 低层 π0/action expert 保持原生控制频率和 action horizon。
- 技能分支按真实时间戳重采样到初始 `10 Hz`，而不是直接按帧编号切段。
- M0 审计后以秒锁定 `τ_min`、`τ_max`；初始搜索范围为 `0.25–4.0 s`，不是论文结论。
- 每次仅执行预测 action chunk 的前缀 `h_exec`，随后重新观测并检查终止。`h_exec` 初始取模型 horizon 的 `1/4`，并受安全控制频率约束。

这样既避免 50 Hz 控制噪声主导边界，也不降低低层闭环精度。

### 3.3 对象中心状态 `x_t`

原始因果观测优先构造：

\[
x_t^{obs}=[T_{O}^{E},\ \dot T_{O}^{E},\ q_t,\dot q_t,\mathrm{gripper}_t,
\mathrm{contact}_t,\ v_t^{img},\ m_t],
\]

其中 `T_O^E` 是末端在当前操作对象坐标系中的 SE(3)，`v_t^{img}` 是局部视觉 patch/关键点特征。旋转使用连续 6D 表示，所有差分按真实时间间隔归一化。目标关系不属于原始观测：`g_k` 只能来自时刻 0 已显式给定的目标，或由因果子目标头预测；演示 segment 末状态只能作为监督/单独 oracle，不能回填为部署输入。

为防止离散 mode 读取任务/工位捷径，posterior 内部进一步拆成：

\[
x_t^z=[\Delta T_O^E,\dot T_O^E,\Delta\mathrm{gripper},
y_t^{contact/event},\Delta v_t^{img},M_{sensor}],
\]

\[
x_t^r=[x_t^z,T_O^E,q_t,\dot q_t,g_k^{provided/predicted},M_{feature}].
\]

`z` 分支只读 `x^z` 和以段初为原点的 canonical action 变化，不读 `q_t`、绝对工位或目标；`r`/控制分支可读 `x^r`，以保留实例几何和控制构型。演示末状态产生的 `g_k^{target}` 只进入 loss，不进入 `x^z/x^r` 的在线调用链。

对象选择按以下优先级进行：数据集对象元数据/仿真真值 → 已标注检测框或关键点 → 语言条件视觉注意区域。若真实数据没有可靠对象姿态，则将几何 outcome 主张降级为“视觉状态后果”，不可用隐藏真值替代部署输入。

绝对工作空间位置不进入离散 tokenizer；它可进入连续 residual prior 和低层控制器。这一分离用来防止 `z` 记住某个任务的固定工位，同时保留实际执行所需坐标。仅“不输入 task ID”并不足够，评测还要对 `z/r` 分别做 task probe 和 outcome/contact 条件 task probe。

## 4. 事件边界与在线终止

### 4.1 事件特征

每时刻事件特征为：

\[
e_t=[\Delta \mathrm{gripper},\|v_E\|,\Delta \hat v_E,\mathrm{jerk},
\Delta T_O^E,\Delta v_t^{img},\Delta f_t,\mathrm{role}_{arms},m_t].
\]

接触状态与接触事件分开建模，避免把持续状态和瞬时转换混成一个分类头：

- `contact_state ∈ {free, contact, constrained}`；
- `contact_event` 为 `{establish, release, slip, impact}` 的多标签向量；全零表示无事件，允许同一采样窗口内同时出现建立接触与冲击。

双臂扩展为每臂一组状态/事件，并可选记录协作 gate；它不是同一本体核心实验的必需模块。没有力觉时，接触标签标为 proxy，并保存来源与置信度。

### 4.2 两个边界模型而不是一个

1. **离线 boundary posterior** `B_φ(e_{1:T})`：逐帧 stem 后接双向 temporal tower，用于生成/修正训练分段。
2. **在线 termination head** `H_ψ(c_t^0,h_t^{evt},z_k,r_k,g_k,Δt_k)`：`h_t^{evt}` 来自逐帧 stem 后的独立 causal event tower，`c_t^0` 是因果视觉/本体上下文，`Δt_k` 是当前技能已执行时长。

二者只共享不跨时间的逐帧投影 stem，不共享 temporal attention、卷积状态或归一化统计。在线速度、jerk、视觉/力差分只使用后向差分和因果滤波；中心差分仅可用于离线标签生成。离线边界不能直接用于部署，否则会产生未来信息泄漏。

在线终止用离散时间 hazard 表示：

\[
h_t=P(b_t=1\mid b_{<t}=0,c_{\le t},z_k,r_k,g_k),\quad
P(T=t)=h_t\prod_{j<t}(1-h_j).
\]

### 4.3 弱标签和分段规则

边界弱标签来自夹爪转换、接触状态转换、速度驻点、对象开始/停止随动、双臂角色切换。各来源单独保存，不先合并成不可追踪的单一标签。训练用置信度加权 focal BCE；预测后执行：

1. `τ_min` 内禁止新边界；
2. 对候选概率做局部 NMS；
3. `τ_max` 到达后产生 `timeout` 截断，但不把它标为正常语义边界；
4. 相邻同类短段可在 outcome/contact signature 相同且合并损失下降时合并。

每个截断保存 `boundary_reason ∈ {event, episode_end, timeout, manual}`。`timeout` 只表示删失/续段，不能监督正常 termination，也不能作为新的 skill 语义；下一窗口延续原 `z`，或在安全规则触发时重规划。

termination 监督映射固定如下：`event/manual` 为正常终止正例；段内有效时刻为 survival 负例；`episode_end` 只有在成功标签或人工确认自然结束时为正例，否则按删失；`timeout` 与安全/故障中断均按删失，并由独立 abort/safety 状态处理。hazard NLL 不把删失时刻记作正例。

边界损失：

\[
\mathcal L_{boundary}=\mathcal L_{focal}^{weak}
+\lambda_d\mathcal L_{duration}
+\lambda_s\mathcal L_{sparsity}
+\lambda_{stab}\mathcal L_{stability}.
\]

`L_stability` 要求对轻微时间裁剪、图像增强和状态噪声的边界位置在容差内一致。少量人工边界只用于校准和 F1 评测；oracle 分段单独作为上界。

### 4.4 为何采用先分段、后交替更新

一开始端到端学习边界、token 和动作策略会出现多重等价解：多切、少切都可能靠 decoder 补偿。主流程先用事件弱监督获得可审计分段，再冻结边界训练 tokenizer。NMS、最短/最长时长和 merge 都是硬操作，默认实现**不声称可微端到端联合优化**。

若后续需要互相改进，采用可审计的交替流程：固定 `B_φ` 生成分段 → 训练 tokenizer/decoder → 根据验证集事件误差更新 `B_φ` → 重新生成训练分段。action loss 不穿过硬 NMS。soft pooling/Gumbel boundary 只能作为单独实验，不能与默认实现混写。这比直接宣称“完全无监督技能发现”更可复现，也保留了核心研究问题。

## 5. 混合技能 tokenizer

### 5.1 Segment encoder

段编码器使用两个输入受限的分支，初始均为 4 层 temporal Transformer、`d_s=256`、8 heads：离散分支 `E_z` 只读 `x^z` 与相对段初的 canonical action 变化；实例分支 `E_r` 读取 `x^r`、原生/canonical 有效动作和离散分支特征。两者均不输入语言、任务编号或场景编号，`E_z` 额外禁止目标和绝对关节/工位。

段表示由 attention pooling 获得：

\[
u_k^z=E_z(x^z_{t_k:t_{k+1}},\Delta a^{canon}_{t_k:t_{k+1}},M),
\quad
u_k^r=E_r(x^r_{t_k:t_{k+1}},a_{t_k:t_{k+1}},M,u_k^z).
\]

### 5.2 离散共享 mode `z_k`

主实现采用 FSQ：

\[
z_k=Q_{FSQ}(W_z u_k^z),\quad d_z=4,\quad L=(3,3,3,3),
\]

理论组合数为 81。该数只作为初始容量，M0/M1 用活跃组合数、held-out 性能和 collapse 指标决定是否调整。选择 FSQ 而不是将 EMA-VQ 作为主实现，原因是：

- 无需学习 codebook，减少小数据下 dead code/collapse 的不稳定性；
- 高层先验可对四个离散坐标分别输出三分类，训练与采样简单；
- 仍可将完整四元组视作可交换 mode，进行因果 skill swap。

EMA-VQ `K=32` 保留为量化器对照。FSQ 的均匀使用率不被当作技能语义证据。

### 5.3 连续实例 residual `r_k`

后验为：

\[
q_φ(r_k|u_k^r)=\mathcal N(\mu_k,\operatorname{diag}\sigma_k^2),\quad r_k\in\mathbb R^{16}.
\]

`r` 保存对象尺寸、准确目标位姿、速度/力度与轨迹微调，不承担共享 mode 的主要语义。为避免 `r` 绕过离散瓶颈：

- 维度初始限制为 16；
- 训练时对 `r` 使用 20% whole-vector dropout；
- 使用与在线先验的 KL/容量约束；
- 粗粒度 outcome/contact head 只读 `z` 与段初状态；精确末态/力回归 head 读取 `z+r`，给 `r` 明确学习信号；
- 分别报告 `z-only`、`r-only` 和 `z+r` 的 probe/rollout。

### 5.4 在线先验

在线预测按“先子目标、后技能”执行，避免循环依赖。先用不含未来子目标的上下文

\[
c_k^0=E_{ctx}(I_{\le t_k},p_{\le t_k},l,h_{<k})
\]

预测 `g_k` 与 feasibility；再把可达子目标写回技能上下文：

\[
c_k=E_{skillctx}(c_k^0,g_k).
\]

其输出为：

\[
p_ψ(z_k|c_k)=\prod_{j=1}^{4}\mathrm{Cat}(z_{k,j};\pi_{k,j}(c_k,z_{k,<j})),
\]

\[
p_ψ(r_k|c_k,z_k,g_k)=\mathcal N(\hat\mu_k,\operatorname{diag}\hat\sigma_k^2).
\]

历史 `h_{<k}` 包含最近的 `z`、持续时间、观测 outcome、termination/boundary reason 和是否安全回退；不保留原始 task ID。`r` 的完整数值默认不跨技能传递，只保留是否发生大幅纠偏的摘要，避免 prior 通过历史 residual 记忆整条轨迹。

posterior-prior 对齐损失为离散坐标交叉熵与高斯 KL。蒸馏目标显式 stop-gradient：

\[
\mathcal L_{prior}=\mathrm{CE}(\operatorname{sg}[z^q],p_ψ(z|c))
+D_{KL}(\operatorname{sg}[q_φ(r|segment)]\|p_ψ(r|c,z,g)).
\]

训练还会随机截取段内 `25%/50%/75%` 的因果前缀。对每个位置 `j`，posterior 从剩余 suffix `segment_{j:end}` 编码 residual-to-go，prior 只看截至 `j` 的因果前缀与重新表达的剩余子目标：

\[
\mathcal L_{r-prefix}=\sum_j D_{KL}(\operatorname{sg}[q(r_j|segment_{j:end})]\|p(r_j|prefix_{\le j},z,g_j))
+\eta\sum_{j>1}\|\hat\mu_j-\hat\mu_{j-1}\|_2^2.
\]

这样部署时重观测后更新 `r` 有对应训练分布且不会无约束跳变。新任务 few-shot 时优先只更新 prior、residual adapter 和 LoRA，不更新 tokenizer，以检验是否真的复用了技能。

### 5.5 M1 轻量动作 decoder

在接入 π0 前，用统一的 4 层 conditional flow decoder `D_mvp(x_{t_k},z_k,r_k,g_k)` 重建归一化原生 action segment。对噪声 `ε` 和 `τ∈(0,1)`：

\[
a_τ=(1-τ)ε+τa,\qquad
\mathcal L_{mvp-flow}=\|M_{action}\odot(D_{mvp}(a_τ,τ,x,z,r,g)-(a-ε))\|_2^2.
\]

该 loss 通过 FSQ straight-through 和 `r` 的 reparameterization 更新两个 posterior 分支；`L_capacity` 只限制 `r` 容量，不能替代动作学习信号。discrete-only、continuous-only、`z+r` 使用相同 decoder、训练步数、action horizon 和搜索预算。M1 结果只证明 latent 瓶颈可行性，正式策略性能仍以 π0/强基线的 prior-only rollout 为准。

可变时长段在 native control rate 上右侧 padding 到

\[
H_{max}=\lceil τ_{max}f_{control}\rceil,
\quad M_{action}=M_{time-action}[:,:,None]\land M_{native-dim}[:,None,:].
\]

`M_action` 形状为 `[B,H_max,D_a]`，flow loss 除以 `max(sum(M_action),1)` 而不是固定张量大小。短段只 padding、不时间拉伸；超过 `H_max` 的原始段以 `timeout/censored` 续段，不能截断后伪装成正常终止。不同 control rate 的本体不混在同一 M1 batch；核心同一本体实验因此保持一个 native rate。

## 6. Grounded subgoal

### 6.1 子目标内容

`g_k` 不是最终任务图，也不是自由文本，而是当前技能可达的短时对象关系。几何和视觉是带 modality mask 的并列 head，不要求所有数据都存在：

\[
g_k=\mathrm{Fuse}([hat g_k^{geo},M_{geo}],
[hat g_k^{vis},M_{vis}],[\hat g_k^{contact},M_{contact}]).
\]

`g^{geo}` 为当前到 segment 末的相对 SE(3)/对象关系，`g^{vis}` 为冻结视觉编码器的 128 维局部未来 latent，`g^{contact}` 为目标接触状态/事件。几何真值缺失时只关闭相应 loss，不以全零向量冒充目标。训练标签可来自真实 segment 末状态，但**只作为监督 target**；主训练进入低层的条件按第 9 节从 teacher-forced `g_target` 逐步切换到因果预测 `g_pred`。验证/rollout 的完整部署链只能用 `g_pred`，真实末状态或未来视觉 latent 仅用于 oracle 上界。

子目标 proposal head 使用 `K_g=3` 个 learned query，每个 query 分别输出 `[B,D_geo]`、`[B,128]`、`[B,7]`、proposal score 与 feasibility，其中 contact 为 3 个状态 logit 与 4 个事件 logit；各自投影到 `d_g=256`。候选 `j` 的融合定义为

\[
g_k^{(j)}=\operatorname{LN}\left(
\frac{\sum_i M_i W_i\hat g_i^{(j)}}{\max(\sum_i M_i,1)}
+\sum_i E_i(M_i)\right),\quad i\in\{geo,vis,contact\}.
\]

缺失模态由 `M_i=0` 屏蔽，并通过 mask embedding 告知低层；若三种目标全缺，feasibility 强制为 0，不执行低层。`D_geo` 由数据契约固定，只包含部署时可定义的对象关系；隐藏真值维不能临时加入。训练用 best-of-K target loss、proposal score CE 和候选 diversity margin；部署接口明确为 `G_ψ(c_k^0)={(g_k^(j),score_j,ρ_j)}_{j=1..K_g}`，再逐候选调用 `p(z|c,g^(j))`，因此“重选”不依赖一个未定义的确定性 top-K。

### 6.2 可达性约束

子目标头同时预测 `g_k` 和 feasibility `ρ_k`。正样本是演示中在该段时长内真实到达的目标；负样本由同场景过远位姿、碰撞目标、错误接触模式和跨对象错配构造。

\[
\mathcal L_{subgoal}=\mathcal L_{SE(3)}+\mathcal L_{visual}
+\mathcal L_{contact}+\mathcal L_{reachability}.
\]

推理时若 `ρ_k` 低于阈值，不将不可达目标交给低层，而是请求重选子目标/技能。报告 final-goal oracle 和 intermediate-subgoal oracle，区分规划误差与控制误差。另做 `2×2` 主消融：无/有 subgoal × 无/有 skill，防止把 SuSIE 式中间目标收益误归因给 tokenizer。

失败状态机固定为 `PROPOSE → FEASIBILITY_CHECK → EXECUTE → VERIFY`。低 feasibility 时从冻结 subgoal proposal head `G_ψ` 的 `K_g=3` 个候选中排除已尝试项，选择下一个满足安全约束的候选，再据该 `g` 预测 `z,r`；最多重选 `N_replan=2` 次。若候选耗尽、连续两次状态变化低于任务阈值、timeout 后仍无进展或安全监控触发，则进入 `SAFE_STOP`，不得无限重采样。`K_g/N_replan/状态变化阈值` 在 M0 pilot 后写入协议并冻结。

## 7. π0 低层执行器接入

### 7.1 条件注入

技能条件为：

\[
s_k=\mathrm{MLP}([\mathrm{emb}(z_k),r_k,g_k,y_t^{observed},m_k]).
\]

`g_k` 已含模型预测的目标接触分量，`y_t^{observed}` 是当前可观测接触/代理状态；不再重复拼接一份目标 contact，低层输入也不使用未来真值接触标签。

当前已核验基座为相邻目录 `../openpi` commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`，JAX 接口位于 `src/openpi/models/pi0.py::embed_suffix`。采用两种轻量注入，并将各自单独消融：

1. 在 `Pi0.embed_suffix` 的 action token 前加入 `mode token` 与 `instance token`；
2. 在 action expert 每隔若干层加入 residual adapter `H'=H+W_{up}σ(W_{down}s_k)`；`W_down` 随机初始化、`W_up` 零初始化，避免同时把缩放和权重置零造成死分支。

π0 在 state token 与 action tokens 之间插入 skill tokens；π0.5 则在 action tokens 前插入，并可将 skill projection 以门控加法并入已有 `adarms_cond`。两个 skill tokens 属于同一 attention block：π0 的 suffix `ar_mask` 固定为 `[True(state), True(skill_1), False(skill_2), True(action_1), False×(H_a-1)]`；π0.5 为 `[True(skill_1), False(skill_2), True(action_1), False×(H_a-1)]`，`input_mask` 为对应长度的有效 token mask。主干 PaliGemma/SigLIP 初期冻结，action expert 使用原权重或小学习率；零初始化 up-projection 使初始行为等价于原 π0。若 token-only 已达到同等收益，则删除 adapter，优先选择更简单实现。

### 7.2 动作空间和 mask

- skill encoder 可以读取已有 128 维 canonical state/action 与布尔 mask；
- 已核验 openpi checkpoint 的动作张量接口为固定 `D_π=32`。核心实现不重建 checkpoint projection：数据适配器把原生 `D_a≤32` 动作装入已配置槽位，生成 `M_π [B,H_a,32]`，模型输出后按同一映射 unpad 回原生动作；
- `CASKPi0` 在训练时将 `actions/noise/x_t/u_t` 全部逐步乘 `M_π`，在 `action_in_proj` 前保证无效槽为零；`compute_loss` 在原实现 `mean(axis=-1)` 之前应用 `M_π`，按有效元素归一化；
- 采样 ODE 初始化 noise、每一步输入 `x_t`、预测 `v_t` 和更新后的 `x_{t-dt}` 都乘 `M_π`，防止无效槽经 action projection/下一步 ODE 回灌有效预测；最终再 unpad 到原生动作；
- 若本体原生动作不能无损映射到 32 维，则需要重建 input/output projection 并单独报告 checkpoint 兼容性损失，不进入首轮核心实验；
- `M_feature` 在投影前屏蔽无效 canonical 维并作为显式特征输入，相关 loss 同样屏蔽；只有 `M_time` 能进入 attention key-padding，二者不能混用。

这样做是为了隔离“共享语义表示”和“不同控制坐标”。若强制 π0 直接输出 128 维 padded action，性能下降可能只是接口污染，而不是技能机制失败。

### 7.3 闭环执行

每个决策周期执行：

```text
观测 → 子目标 g → 先验选择 z,r → π0 生成 action chunk
     → 只执行前 h_exec → 更新观测
     → termination / safety / feasibility 检查
     ├─ 未终止：保持 z，更新 r，继续当前技能
     ├─ 正常终止：选择下一技能
     └─ timeout/异常：回退、重规划或安全停止
```

`z` 在一个 segment 内保持不变；每次重新观测后，以固定的 `z,g` 和新上下文再次计算同一个 residual prior 来更新 `r`，不额外引入未定义的未来编码器。这样共享 mode 稳定，同时允许接触后的毫米级纠偏。

## 8. 损失函数与梯度路径

总损失：

\[
\begin{aligned}
\mathcal L ={}& \mathcal L_{flow}
+\lambda_p\mathcal L_{prior}
+\lambda_q\mathcal L_{quant}
+\lambda_o\mathcal L_{outcome}
+\lambda_c\mathcal L_{contact}\\
&+\lambda_x\mathcal L_{cross-task}
+\lambda_b\mathcal L_{boundary}
+\lambda_t\mathcal L_{termination}
+\lambda_g\mathcal L_{subgoal}
+\lambda_i\mathcal L_{invariance}
+\lambda_r\mathcal L_{capacity}.
\end{aligned}
\]

### 8.1 各损失定义

- `L_flow`：M1 使用第 5.5 节统一 decoder，M4 使用 π0 原生 flow matching；都只在有效动作维计算。
- `L_prior`：在线先验对 detached posterior `z` 的分类损失、`KL(sg[q(r)]||p(r))` 与 `L_r-prefix`；蒸馏时不把弱 prior 反向拉坏 posterior。
- `L_quant`：主实现 FSQ 不需要 codebook/commitment loss，因此此项置零并只监测饱和率与活跃组合；VQ 对照才使用 codebook 与 commitment loss。
- `L_outcome`：粗 head 用 `z+段初状态` 预测 canonical outcome bin/方向并更新 `E_z`；精确 head 用 `sg[z]+r+段初状态` 回归连续对象 delta，只将精确残差梯度送入 `E_r`，使用 Huber + rotation geodesic loss；同时做 state-only 和 shuffled-`z` 对照，报告 `z` 的条件增量预测力。
- `L_contact`：分别预测 contact state 与 contact event；粗 head 读 `z` 并更新 `E_z`，精确力/峰值 head 读 `sg[z]+r` 并更新 `E_r`，均按传感器/代理置信度加权。
- `L_cross-task`：只在不同 task 间构造 supervised contrastive pair，作用于 FSQ straight-through 输出而不是 `r`。
- `L_boundary`：第 4 节的弱监督、时长、稀疏和稳定性损失。
- `L_termination`：在线因果 hazard 对离线校准边界的 survival NLL；timeout/censored 段按删失样本处理，不伪造正常终止标签。
- `L_subgoal`：相对位姿/视觉未来/接触/可达性损失。
- `L_invariance`：gradient reversal task classifier，只作用于 `z` 前的离散分支。
- `L_capacity`：`KL(q_φ(r|segment)||N(0,I))`，对 posterior 有梯度并配合 dropout；它与只更新 prior 的蒸馏 KL 分开。

### 8.2 跨任务正负对

正对首先是**带置信度的候选对**：不同 task、对象角色与初态可兼容、约束/安全接触方式一致、contact state/event signature 相同，且对象中心 outcome 在传感器容差内一致。不确定样本不进入 contrastive loss；一部分高置信 pair 必须人工审计。负对优先选“动作轨迹相似但 outcome/contact/约束方式不同”的 hard negative，防止模型只按速度形状聚类或把不同安全路径强行合并。

pair 标签只能来自仿真状态、外部跟踪、接触传感器或经审计的代理事件，不能由当前 `z` 自己生成，否则会形成伪标签闭环。

令 `\tilde z_i=Q_{FSQ}(W_zu_i^z)` 使用 straight-through，`v_i=normalize(P\tilde z_i)`，则

\[
\mathcal L_{cross-task}=-\sum_i\frac{1}{\sum_{p\in P(i)}w_{ip}}
\sum_{p\in P(i)}w_{ip}\log
\frac{\exp(v_i^Tv_p/τ_c)}{\sum_{a\ne i}\exp(v_i^Tv_a/τ_c)}.
\]

置信度 `w_ip` 和 pair 集合 stop-gradient。该损失只更新 `P,W_z,E_z`；不更新 `E_r`、action decoder 或 pair 生成器，因此能直接塑形离散 mode 而不把实例 residual 拉成跨任务一致。

只有 `P(i)` 含至少一个高置信跨任务正对的 anchor 才进入求和，空正对 anchor 直接跳过。batch sampler 优先放入至少两个预先审计的 compatible pair；若当前 batch 无有效 pair，则 `L_cross-task=0` 并记录计数，禁止复制样本或用同任务伪正对避免除零。

### 8.3 梯度隔离

- 语言和 task ID 的梯度不进入 posterior tokenizer；
- 只有粗粒度 outcome/contact 语义 head 不读取 `r`；精确末态/力 head 读取 `z+r`，在不迫使 `z` 几何碎裂的同时给 `r` 学习信号；
- `L_invariance` 在 outcome/contact 预测已稳定后再 warm-up，并设置小权重；
- 若 task 与真实 outcome 高度相关，不能强行把 task probe 压到随机水平，否则会删除有效信息；最终以 held-out transfer 而不是对抗损失值选模型。

### 8.4 条件来源、梯度和冻结规则

| 阶段/路径 | 低层条件 | 梯度更新 | stop-gradient/离散规则 |
|---|---|---|---|
| M1 tokenizer | posterior `z^q,r^q` + target `g` | `E_z,E_r,D_mvp` | FSQ straight-through；capacity KL 更新 `q(r)` |
| M2 boundary | 固定分段内 posterior | boundary 单独训练；tokenizer 按冻结分段训练 | action loss 不穿过 NMS/merge |
| M3 prior | 预测 `g`，teacher-forced `z^q` 训练自回归坐标，`r^q` 作目标 | `E_ctx,subgoal,p(z),p(r)` | `z^q,q(r)` 目标全部 detached |
| M4a executor warm-up | detached posterior `z^q,r^q`；target/predicted `g` 混合 | skill tokens/adapter、action expert | tokenizer 冻结 |
| M4b scheduled prior | `g,z,r` 的 predicted 比例按 `0→50%→100%` | executor、prior/LoRA | `z` 用 argmax/采样且 flow 不穿过 categorical；`p(z)` 由 CE 更新；连续 `r` 可接 flow 梯度 |
| 主验证/rollout | 100% predicted `g,z,r` + causal termination | 无 | 禁止调用 trajectory encoder、未来 target 或双向 boundary |

M4 每个训练 batch 记录条件来源，不能把 oracle 和 predicted 样本混在同一结果标签下。若 scheduled prior 造成坍缩，回退到更慢的 curriculum，而不在主评测恢复 posterior 条件。

### 8.5 初始权重策略

先将每项损失按有效元素数和经验标准差归一化，再以 `L_flow=1` 为基准。初始建议：`λ_p=1`、`λ_o=0.5`、`λ_c=0.25`、`λ_x=0.25`、`λ_b=0.5`、`λ_t=0.5`、`λ_g=0.5`、`λ_i=0.05`；`λ_r` 采用 KL warm-up。它们是工程起点，不是固定研究结论。若辅助损失梯度范数持续超过 action loss 两倍，使用 gradient clipping/GradNorm，而不是直接以训练总损失最低选模型。

## 9. 分阶段训练协议

### Stage 0：数据与直接策略基线

1. 在任何切窗、分段、重采样或增广之前按原始 episode 拆分；同一采集 session、近重复轨迹、对象实例和场景必须成组。
2. held-out 明确分为：未见技能组合、未见对象/几何、真正新技能负迁移集，三者不可混成一个平均数。
3. 目标任务的 few-shot adaptation demos、调参 validation episodes、最终 rollout seed/场景相互隔离。
4. normalization、事件阈值、proxy-contact 标定和 pair mining 只拟合训练集。
5. 输出带 episode/source 哈希的 split manifest 并冻结；后续实验只引用 manifest 版本。
6. 审计 FPS、动作维、传感器、成功标签和可复用阶段；π0 direct FT 为必做主基线，ACT/Diffusion Policy 至少选一项作为不同策略族的强直接基线。

输出：不可变 split manifest、字段统计、事件信号覆盖率、直接策略结果。

### Stage 1：固定窗混合瓶颈

1. 冻结视觉 encoder；用固定 8/16/32 帧或等秒长窗口训练 `z+r` tokenizer 和第 5.5 节统一 decoder。
2. 完成 discrete-only、continuous-only、`z+r` 三组比较，并加入 QueST/STAR 风格离散瓶颈和 CLAM/OPAL 风格连续 latent 对照；latent bitrate、decoder 容量、训练算力与超参搜索预算匹配。
3. 是否进入下一阶段完全按 G1 判定；未通过不得以“优于某个较弱对照”继续保留离散 mode 主张。

输出：最小可行性结论，不接入完整 VLA。

### Stage 2：事件边界

1. 从多源事件生成可追踪弱标签；标注少量 boundary validation set。
2. 训练离线 boundary posterior 与在线 termination head。
3. 与固定窗、随机边界、直接事件阈值/规则、CompILE/TACO 风格 learned boundary（去掉 contact/outcome 事件特征）、PRISE/BPE、事件切分但无 skill code、oracle 边界比较；补一个 UniVLA 风格约 1 秒固定 token 对照。

是否进入联合模型完全按 G2 判定；仅 boundary F1 或稳定性提高不足以通过。

### Stage 3：在线先验与 grounded subgoal

1. 冻结/半冻结 tokenizer，按第 8.4 节训练 `p(z,r|context)` 对齐 detached posterior。
2. 训练子目标和 reachability head；低层条件从 target `g` 逐步切到 predicted `g`。
3. 强制使用完整因果链 validation，记录 posterior/oracle subgoal 与 predicted chain 的差距。

### Stage 4：π0 skill conditioning

1. 冻结 VLM prefix，以 detached posterior 条件 warm-up 新增 token/adapter 和 action expert。
2. 按 `0→50%→100%` curriculum 混入 predicted `g,z,r`，再对 action expert/LoRA 使用小学习率联合微调；categorical `z` 不接 flow 梯度。
3. 完成原 π0、参数匹配但无 skill 的 adapter、skill token-only、token+adapter，以及 π0+subgoal/π0+skill 的 `2×2` 对照。

### Stage 5：few-shot 迁移

对每个 held-out 任务使用 `1/5/10/20/50` demos。完整 demo-budget 曲线首先做 `2×2`：direct π0、subgoal-only、skill-only、subgoal+skill；再分别比较仅更新对应 prior、prior + 低层 residual adapter、LoRA 和 full FT，tokenizer 默认冻结。skill-only 的 prior 不接 `g`，subgoal-only 不生成/输入 `z,r`。四组匹配优化步数和调参预算；通过调整 active adapter 宽度将可训练参数控制在主模型的 ±5%，不能做到时增加单独的 parameter-matched adapter control 并报告实际参数量。每个预算使用相同抽样清单；预训练数据量单独报告，不能把“目标任务少样本”表述为“总数据少”。`prior-only rollout` 指条件来源 100% predicted，不是一个适配参数组名称。

## 10. 评测与因果检查

### 10.1 主结果

- 成功率及 95% seed—demo subset—rollout 分层 bootstrap 区间；Wilson 只作为单个固定策略/固定数据子集条件的补充；成功、任务超时、安全停止和阶段失败由每任务 manifest 预先定义；
- demo-budget 曲线和以 `log2(1+demos)` 为横轴的归一化 AUC；AUC 用 demo 子集、训练 seed 和 rollout 的分层 bootstrap；
- 达到预注册成功率阈值所需 demonstrations；
- 末端位置/角度误差、插入深度、峰值力、滑移、重抓和完成时间；
- 分阶段成功率和失败类型。

M0 pilot 后、最终 held-out test 之前冻结 `experiment_protocol.yaml`：主指标、主基线、最小效应/非劣界、训练 seed 数、demo 子集重复数、rollout 数、boundary F1 时间容差、bootstrap 单位、probe 模型容量/split/条件匹配基线，以及 swap source-target 清单都写入并记录哈希。rollout 数由预期效应和功效分析确定，不能看到最终结果后再增删。起始下限为 3 个训练 seed、每预算 5 个 demo 子集；真机条件若资源不足，必须报告实际功效与宽置信区间。

若使用论文名方法的官方代码与关键训练配方，标为“official reproduction”；若只实现同类结构，统一标为“X-style reimplementation”，列出缺失组件，不能直接以原论文方法名暗示忠实复现。

### 10.2 技能本体证据

- posterior/prior 的 `z → outcome/contact` probe；
- posterior/prior 的 `z → task ID`、`r → task ID` 及 outcome/contact 条件 task probe；
- 跨任务检索 purity/NMI，仅作为辅助证据；
- `z`、`r` 的独立与联合动作/后果 probe；
- boundary F1@时间容差、平均段长、边界扰动稳定性；
- 活跃 FSQ 组合数和熵，但不把高熵等同于好技能。

### 10.3 Skill swap

对一个源 segment 取 `z_src`，在目标上下文重新预测 `r_tgt`，并执行 `π_L(a|z_src,r_tgt,c_tgt)`：

- compatible swap：相同 outcome/contact、不同 task；
- incompatible swap：contact 或 outcome 明确冲突；
- random-code control；
- `z+r` 整体交换对照，用来证明实例 residual 不应直接跨对象复制。

若只有整体交换有效而“固定 `z`、重算 `r`”无效，说明 `z` 没有承载共享 mode；若 compatible 与 incompatible 同样成功，说明交换任务设计不具诊断性。

`z_src` 只能来自训练 split 建立并冻结的 source library；不能根据目标 rollout 的未来结果反选 source。compatible/incompatible 定义、source-target 对和随机/置换基线在 rollout 前由 train/validation 信息确定，写入协议并记录哈希。最终 test 只执行冻结清单。

### 10.4 必做消融矩阵

| 因果问题 | 对照 |
|---|---|
| 事件边界是否必要 | 固定时长、UniVLA 式约 1 秒 token、PRISE/BPE、随机、直接事件规则、无事件特征 learned boundary、事件边界但无 skill、CASK、oracle |
| 离散 mode 是否必要 | QueST/STAR 风格 discrete-only、CLAM/OPAL 风格 continuous-only、`z+r` |
| 语义来自 outcome/contact 吗 | action-only、+outcome、+contact、二者同时 |
| task invariance 是否有益 | 无 GRL、GRL、只做跨任务配对 |
| 子目标与技能各自贡献 | π0、π0+subgoal、π0+skill、π0+subgoal+skill；另报 final/oracle intermediate |
| π0 条件注入是否只是参数增加 | 原 π0、等参数无 skill adapter、skill token、token+adapter |
| 接触信号是否可靠 | oracle contact、真实传感器、proxy、无 contact |

## 11. 设计决策记录

| 决策 | 选择 | 证据来源/主张身份 | 为什么 | 推翻条件 |
|---|---|---|---|---|
| 核心范围 | 同一本体多任务 few-shot | 报告范围控制；研究边界 | 跨本体有强替代解释且工程范围过大 | 同一本体无重复技能，但多本体数据充分 |
| 语言层 | 冻结/外部接口 | RT-H/GPLA 风险；工程隔离 | 避免把 skill 收益与 VLM 规模混淆 | 低层已成立且研究问题转向语言泛化 |
| 坐标表示 | 对象中心输入离散分支 | UHAS/Cloak 等几何捷径反证；工程支撑 | 减少 task/location shortcut | 无法可靠获得对象关系且视觉 latent 更稳定 |
| 分段 | 事件弱监督 + 因果 termination | 弱监督/可变边界有 CompILE/TACO 等先例；其与跨任务语义的组合才是主张 | 可审计、符合接触技能定义 | 事件边界不改善下游指标 |
| 离散量化 | FSQ 4×3 levels | 继承 FSQ/VQ 路线；非创新 | 小数据更稳定、先验易预测 | VQ 在统一预算稳定显著更优 |
| 技能表示 | `z+r` | STAR/CLAM 等证据；工程支撑 | 同时保留共享模式和精细控制 | continuous-only 全面更优 |
| 语义约束 | outcome/contact + 跨任务 pair | CASK 核心组合主张 | 防止压缩码冒充技能 | 约束不改善 swap/held-out transfer |
| 在线选择 | posterior-prior 蒸馏 | latent policy 常规做法；部署必要 | 杜绝未来泄漏 | 不可推翻；这是部署必要条件 |
| 子目标 | 短时对象关系 + reachability | SuSIE 等启发；工程支撑 | 缩短控制距离、减少不可达目标 | 预测子目标不如无子目标/最终目标 |
| 控制输出 | 本体原生动作维 | CrossFormer/现有 pipeline；工程防护 | 避免 padding/语义差异污染动作头 | 统一输出在严格 mask 下更优且无负迁移 |
| π0 接入 | suffix token + 零 up-projection adapter | openpi 接口；工程支撑 | 保留基座并控制算力 | token-only 已足够则删 adapter |
| task 对抗 | 小权重、后期开启 | 表示去偏常规手段；辅助项 | task/outcome 可能统计相关 | 对抗头损害 held-out performance |

论文核心创新只归于“事件边界 + 跨任务 outcome/contact 语义 + 窄容差 few-shot 因果验证”的组合；`z+r`、subgoal、posterior-prior 和 π0 连续执行均明确作为继承或工程支撑。

## 12. 阶段验收与停止门槛

| Gate | 必须回答的问题 | 通过证据 | 不通过时动作 |
|---|---|---|---|
| G0 数据可行性 | 不同任务是否真实共享阶段？ | 跨任务 outcome/contact 匹配段数量与人工审计 | 重设计任务/采集，停止训练“共享技能” |
| G1 混合瓶颈 | `z+r` 是否兼顾迁移与精度？ | 预注册 few-shot AUC 优于最佳匹配 continuous/flat 主基线，且精细误差满足预注册非劣界 | 放弃离散 mode 主张或缩小场景 |
| G2 事件边界 | 边界是否比固定窗有下游价值？ | 除 boundary F1 外，success/AUC 或 compatible-swap 至少一项超过预注册效应门槛 | 固定窗作为最终模型，不声称边界创新 |
| G3 在线可用性 | 完整因果链能否替代 oracle？ | predicted `g,z,r` + causal termination 的 rollout 达到预注册下界；任何未来调用均为失败 | 改进 prior，不得报告 oracle 为主结果 |
| G4 共享语义 | `z` 是技能还是 task/轨迹码？ | compatible swap、条件增量 outcome/contact probe、outcome/contact 条件 task probe 联合成立 | 删除“共享技能”表述 |
| G5 样本效率 | 是否真正在 held-out 任务节省示教？ | 相对预注册最强主基线的 demo-AUC/阈值差及 CI，三类 held-out 分开并含负迁移 | 论文结论改为表示学习分析 |

任何单一 Gate 不能由 t-SNE、code usage 或训练重建误差替代。Gate 中的数值效应门槛与非劣界必须在 M0 pilot 后写入带哈希的协议，且在最终测试前冻结；当前缺少真实任务分布，本文不伪造这些数值。

## 13. 尚未锁定的工程参数

以下项目必须由真实数据/GPU 状态决定，当前不能从文献报告推断：

- 机器人与原生动作维、控制频率、π0 action horizon；
- 外部/腕部相机数量、时间同步误差；
- 力矩、触觉或电流是否可用及其标定；
- 每任务 episode 数、任务组合和成功标签；
- GPU 型号/数量及允许训练时长；
- 能否获得对象姿态/关键点，或只能使用视觉 latent；
- 真实机器人每个条件允许的 rollout 数。

在这些信息确认前，本文的维度、时长和损失权重均是起始配置；不应写入论文摘要作为既定事实。

## 14. 实现产物建议

后续代码应至少形成以下可独立测试组件：

```text
cask/
  data/object_centric_adapter.py
  data/event_labels.py
  models/boundary.py
  models/segment_encoder.py
  models/hybrid_tokenizer.py
  models/skill_prior.py
  models/subgoal.py
  models/pi0_skill_adapter.py
  policies/cask_policy.py
  eval/boundary_metrics.py
  eval/skill_swap.py
  eval/demo_budget.py
  configs/cask_m1.yaml
  configs/cask_pi0.yaml
```

单元测试重点覆盖：

- `M_time/M_feature/M_sensor` 语义分离，feature padding 不影响 latent/loss；
- 任意改变 32D 无效动作槽的外部噪声，在逐步 mask 后不改变 native action 输出；每个采样 ODE 步后无效槽严格为零；
- posterior 的 `z` 分支不读 task/language/目标/绝对关节，主低层调用不读 segment 末状态或未来视觉 target；
- 任意修改时刻 `t` 之后的后缀，时刻 `t` 的 termination 与在线 event feature 完全不变；
- 完整部署推理不调用 trajectory encoder、双向 boundary、oracle subgoal 或未来 contact；
- skill swap 只替换指定 `z`，并在目标上下文重算 `r`；
- split 在切窗前完成，session/对象/近重复 episode 不跨 split；normalization 与 pair mining 不读取 validation/test；
- timeout 保留原 `z` 或触发安全重规划，不生成正常语义 termination 标签。

## 15. 变更记录

- v0.1（2026-08-14）：从研究报告展开首版模型、训练和评测设计。
- v0.2（2026-08-14）：根据三路独立审查补齐 M1 decoder、posterior/prior curriculum、梯度/stop-gradient、因果边界、输入与 mask 隔离、完整部署链、强基线和严格 Gate。
