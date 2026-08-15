# CASK × π 系列模型：代码构建指南与实时实施日志

> 启动时间：2026-08-14 11:39（Asia/Shanghai）  
> 当前状态：架构与上游接口审计中  
> 设计基线：`model_design.md` v0.2  
> 代码基线：`../openpi` commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`

## 0. 本文如何使用

本文同时承担两项职责：

1. **实施日志**：按时间记录已经核验的事实、作出的设计决策、风险和下一步；
2. **代码指南**：最终给出从数据契约、CASK 独立模块到 π0/π0.5 接入、训练、推理和测试的可执行蓝图。

文中严格区分三类内容：

- **已核验事实**：已从本地代码、配置或官方来源确认；
- **实现决策**：为 CASK 作出的工程选择，可以通过消融推翻；
- **待确认参数**：依赖真实机器人、数据和算力，不在缺少证据时伪造。

## 1. 实时实施记录

### 2026-08-14 11:39 — 检查点 A：仓库与目标确认

已确认当前 `mrc_research` 只有研究与模型规格文档，没有 Python 包、训练脚本或测试。现有模型不是“重新训练一个 π 模型”，而是在 π 系列 action expert 周围增加一条可审计的 CASK 技能路径：

```text
训练专用、看完整动作段：
segment posterior q(z, r | segment) ──蒸馏目标──┐
                                               │
部署专用、只看当前/历史：                      ▼
observation + language → subgoal g → prior p(z, r | context, g)
                                      │
                                      ▼
                              skill condition s
                                      │
                                      ▼
                           π0 / π0.5 action expert
                                      │
                                      ▼
                     action chunk → 执行前缀 → 重新观测
```

因此代码必须分成两类，不能把它们写成一个能读取未来信息的“大模型 forward”：

- **训练表征模块**：离线边界、segment encoder、FSQ `z`、连续 residual `r`；
- **部署因果模块**：subgoal、skill prior、termination、π action expert 和闭环 policy。

### 已核验的本地上游事实

- `../openpi` 当前 commit 为 `15a9616...`；工作树只有用户已有的 `examples/ur5/README.md` 修改，本项目不得覆盖它。
- 主 JAX 实现在 `src/openpi/models/pi0.py`，配置在 `pi0_config.py`。
- `Pi0Config` 默认 `action_dim=32`、`action_horizon=50`；π0.5 通过 `pi05=True` 使用同一 `Pi0` 类。
- π0 使用连续 state suffix token；π0.5 把 state 放进离散输入，并用 action expert 的 adaRMSNorm 注入 flow timestep。
- `embed_suffix` 是最小侵入的 skill-token 接入点；`compute_loss` 与 `sample_actions` 是动作 mask 必须覆盖的两个闭环。
- 当前 `compute_loss` 在动作最后一维直接做 `mean`。若原生动作维小于 32，CASK 必须在 reduction 前按有效元素归一化，不能让 padding 维进入 loss。
- 当前采样从 `[B,H_a,32]` 高斯噪声开始并逐步做 ODE 更新。动作 mask 必须作用于初始噪声、每步 `x_t`、预测速度和更新结果，而不只是最终 unpad。
- 本地 OpenPI 同时含 π0-FAST，但它是自回归离散 action-token 路线；CASK v0.2 的连续 residual、逐步 action mask 和 flow loss都天然对齐 π0/π0.5 的 flow-matching head。

### 初步实现决策 D1：主基座选 π0.5 flow，π0 作为结构消融

这是当前的工程默认项，不是实验结论。理由：

1. 本地 OpenPI 对 π0.5 的公开训练/推理支持本身就是 flow head；
2. CASK 的 M1 decoder 和最终 action expert 都按 conditional flow matching 设计，接口一致；
3. π0.5 有更长的离散上下文默认值，并把连续 action expert 与 VLM prefix 分工得更清楚；
4. π0 和 π0.5 共享主要实现，容易保持相同数据与评测协议做对照。

π0-FAST 暂不作为 CASK 的首个承载基座，而作为强基线。原因不是它较弱，而是把 `z+r` 连续技能条件接入自回归 FAST action token 会同时改变条件接口和动作生成范式，第一阶段难以归因。

### 待核验

- 官方 π 系列各版本的能力、训练方式、许可证与 OpenPI 支持边界；
- `PaliGemma.llm` 的层级 adapter 注入成本，确定首版是否只做 suffix skill tokens；
- 数据管线 canonical mask 的真实字段名和 batch 形状；
- 当前机器能否运行 OpenPI 的 dummy-model 单测。

## 2. 当前最重要的编码原则

### 原则 1：不要直接修改原始 `Pi0` 语义

首版应新增 `CASKPi0`/组合模块，通过继承或小范围参数化复用已核验的 prefix、checkpoint loader 和 flow sampler。原 `Pi0` 行为必须保留为可重复基线。

### 原则 2：模型接口显式携带四类 mask

```text
M_time       [B,L]          segment/temporal padding
M_feature    [B,L,D_c]      canonical skill feature validity
M_sensor     [B,L,D_s]      sensor presence/confidence
M_action     [B,H_a,D_pi]   native action slot validity
```

四者不可复用同一个变量。只有 token 有效性进入 attention mask；feature/sensor/action mask 分别在数值投影与对应 loss 中使用。

### 原则 3：所有部署 API 都要能证明因果

部署入口只接收当前与历史观测、语言、技能历史和 elapsed time。它不得接收 `segment_end`、未来动作、离线双向边界输出或演示末状态。建议在类型层把 `TrainingSegmentBatch` 与 `PolicyObservation` 分开，而不是依赖调用者“记得不用未来”。

### 原则 4：先 token-only，再决定层内 adapter

首个可运行版本只在 action suffix 前插入 `mode token` 和 `instance token`。它修改面小、可零风险加载基座权重，也能先回答技能条件是否有用。只有 token-only 明显欠拟合时，再实现 action-expert 层内 residual adapter；adapter 的 up projection 必须零初始化以保持初始函数等价。

## 3. 下一检查点

1. 完成 π0、π0-FAST、π0.5 的官方事实对比，避免把 π0.5 的能力误写成代码保证；
2. 审核 OpenPI JAX/PyTorch 两条实现，选择唯一主开发栈；
3. 固化包目录、dataclass、张量契约和最小单测；
4. 在不依赖真实数据的范围内建立 shape/mask/causality 测试骨架；
5. 数据与机器人参数仍缺失时，将不能运行的训练环节明确标为 blocker，而不是填入猜测值。

### 2026-08-14 — 检查点 B：π 系列官方边界与开发栈冻结

官方页面、arXiv API 和本地 OpenPI 代码共同确认：

| 模型 | 生成方式 | 本地实现 | CASK 中的角色 |
|---|---|---|---|
| π0 | conditional flow matching；连续 action chunk | JAX + PyTorch | 结构消融、较简单的连续 state-token 基座 |
| π0-FAST | FAST/DCT+BPE 动作 token；自回归生成 | 仅 JAX 完整支持 | “动作压缩不等于 skill”的强基线 |
| π0.5 | OpenPI 当前只公开支持 flow-matching head；action expert 使用 adaRMS | JAX + PyTorch | CASK Stage 4 主基座 |

已核验论文：

- π0：arXiv 2410.24164，*π0: A Vision-Language-Action Flow Model for General Robot Control*；
- FAST：arXiv 2501.09747，*FAST: Efficient Action Tokenization for Vision-Language-Action Models*；
- π0.5：arXiv 2504.16054，*π0.5: a Vision-Language-Action Model with Open-World Generalization*。

不把任何未出现在当前官方 OpenPI 和可核验论文中的“后续 π 型号”写入代码基线。未来升级必须新增独立兼容性审计，不能只改模型名称。

**开发栈冻结为 JAX/Flax**。原因不是个人偏好，而是本地 OpenPI 的能力差异：JAX 路线已有 LoRA、FSDP、EMA 和 π0-FAST；PyTorch 路线在当前 commit 明确不支持这些能力。CASK 需要低显存 LoRA、参数冻结和多阶段训练，主代码若选 PyTorch 会在后期被迫维护两套不等价训练协议。

### 2026-08-14 — 检查点 C：采用独立扩展包，不直接改上游

本轮检查期间，../openpi 的 pi0.py、pi0_config.py、gemma.py 出现了其他并发任务写入的中文注释改动。它们未改变已观察到的模型逻辑，但属于已有工作，必须保留。

因此 CASK 采用 **overlay package**：

    mrc_research/cask/                 # 本课题代码
    mrc_research/scripts/              # CASK 分阶段训练/评测入口
    mrc_research/tests/                # 独立单测
    ../openpi/                         # 固定上游依赖，不直接承载 CASK 源码

上游行为一律用 commit 15a9616 的 git 对象核对；CASK 通过 import、继承和组合复用 OpenPI。这样能同时获得：

1. π checkpoint 参数路径保持兼容；
2. 原始 π0/π0.5 始终可作为未改动基线；
3. 升级 OpenPI 时可清楚看到接口破坏；
4. 不覆盖用户或其他任务正在进行的上游注释/实验。

## 4. 最终建议：先构建 CASK core，再接 π action expert

不要从复制 pi0.py 开始。正确实现顺序由研究因果问题决定：

    Stage 0  数据契约、split、mask、direct π baseline
       ↓
    Stage 1  固定窗 HybridTokenizer + 小型 flow decoder
       ↓
    Stage 2  离线 boundary + 在线 causal termination
       ↓
    Stage 3  subgoal + online skill prior
       ↓
    Stage 4  CASKPi0/CASKPi05 skill-token executor
       ↓
    Stage 5  held-out few-shot 与 skill swap

如果一开始就把所有模块塞进 π0，失败时无法判断是 latent、边界、prior、subgoal 还是大模型微调造成。Stage 1 的小 decoder 是必要的科学隔离层，而不是临时代码。

## 5. 建议代码目录

    cask/
      config.py                    # 唯一的结构/维度配置来源
      types.py                     # TrainingSegmentBatch / PolicyContext / SkillCondition
      masking.py                   # 四类 mask 和 32D slot pack/unpack
      data/
        object_centric_adapter.py  # 原始轨迹 -> x_obs/x_z/x_r
        event_labels.py            # 可追踪事件弱标签
        segment_dataset.py         # split 后切段；绝不在切段后 split
      models/
        boundary.py                # 双向离线边界
        termination.py             # 独立因果 hazard
        segment_encoder.py         # E_z / E_r
        fsq.py                     # 4×3 levels + straight-through
        hybrid_tokenizer.py        # q(z,r|segment)
        mvp_flow_decoder.py        # Stage 1 小型动作 decoder
        subgoal.py                 # K_g proposals + feasibility
        skill_prior.py             # p(z,r|causal context,g)
        skill_condition.py         # z/r/g/event -> 两个 skill tokens
        pi0_skill_adapter.py       # CASKPi0/CASKPi05
      policies/
        cask_policy.py             # PROPOSE→CHECK→EXECUTE→VERIFY
      losses/
        boundary.py
        representation.py
        flow.py
        prior.py
      eval/
        causality.py
        boundary_metrics.py
        skill_swap.py
        demo_budget.py
    scripts/
      audit_dataset.py
      train_stage1_tokenizer.py
      train_stage2_boundary.py
      train_stage3_prior.py
      train_stage4_pi.py
      rollout_cask.py
    tests/
      test_masks.py
      test_fsq.py
      test_causality.py
      test_pi_skill_tokens.py
      test_policy_state_machine.py
    configs/
      cask_m0.yaml
      cask_m1.yaml
      cask_pi05.yaml
      experiment_protocol.yaml

## 6. 类型与张量契约

### 6.1 训练 batch 与部署 context 必须是不同类型

TrainingSegmentBatch 可以含完整 segment 和未来监督目标：

| 字段 | shape | 用途 |
|---|---:|---|
| z_features | [B,L,D_zin] | 只含对象中心变化、接触/事件、相对动作 |
| r_features | [B,L,D_rin] | 增加实例几何、关节、因果/teacher 子目标 |
| native_actions | [B,L,D_a] | Stage 1 flow target |
| canonical_action_delta | [B,L,128] | 仅供共享 skill encoder |
| time_mask | [B,L] | segment padding |
| feature_mask | [B,L,128] | canonical 有效维 |
| sensor_mask | [B,L,D_s] | 传感器存在性/置信度 |
| outcome_target/contact_target | 各自定义 | 语义监督 |

PolicyContext 只能含当前与历史因果量：

| 字段 | shape | 禁止内容 |
|---|---:|---|
| images/state/language | 当前窗口 | 未来图像/动作 |
| skill_history | [B,K,D_h] | 原始 task ID、未来 outcome |
| elapsed_time | [B] | segment 最终时长 |
| observed_contact/event | [B,D_e] | 目标接触真值 |

SkillCondition 是两条路径的汇合点：

    z_coords       [B,4]       FSQ 坐标，取值归一化到 [-1,1]
    residual       [B,16]      posterior 或 prior 的 r
    subgoal_embed  [B,256]     预测 g 的融合表示
    observed_event [B,D_e]     当前已观测事件
    validity       [B,D_m]     条件来源/模态有效性

只有 SkillCondition 能进入 π action expert。部署代码的类型签名不接受 TrainingSegmentBatch，从接口层阻止 future leakage。

### 6.2 四类 mask

1. time_mask：控制 temporal attention 和可变段 loss；
2. feature_mask：在 canonical 输入投影前逐维清零，并作为额外特征拼入；
3. sensor_mask：给 force/contact loss 加权，缺失不等于测量值为 0；
4. action_mask：原生动作装入 π 的 32 个槽后，屏蔽无效时间和无效维。

动作 flow 的正确归一化为：

    squared_error = (v_t - u_t) ** 2
    numerator     = sum(squared_error * action_mask)
    denominator   = max(sum(action_mask), 1)

不能先对 32 维做 mean 再 mask；否则 7D 机器人会被 25 个无效槽稀释梯度。

## 7. CASK core 的逐模块构建方式

### 7.1 object_centric_adapter.py

该文件只负责确定性数据变换，不含可学习参数。它输出两个受限视图：

    x_z = [
      delta(T_O_E), velocity(T_O_E), delta_gripper,
      contact_state, contact_event, delta_visual_patch,
      sensor_mask
    ]

    x_r = [
      x_z, T_O_E, q, q_dot, causal_or_teacher_subgoal,
      feature_mask
    ]

强制规则：

- x_z 的 schema 中不允许出现 language、task_id、absolute_workspace_pose、q 或目标；
- 所有差分按真实 timestamp 计算；
- rotation 用连续 6D 或明确测试过的 SE(3) 表示；
- g_target 只写入 target 字段，不能回填进部署 context。

### 7.2 segment_encoder.py 与 fsq.py

E_z 和 E_r 使用独立参数的 temporal Transformer。二者都先对 padding 做 attention mask，再做 mask-aware attention pooling。

E_z 输出 u_z，经过线性层得到 4 个标量，再做 FSQ：

    bounded = tanh(project(u_z))
    integer = round(bounded * half_levels)
    z_q = bounded + stop_gradient(integer / half_levels - bounded)

levels=(3,3,3,3) 时每个坐标是 -1/0/1，组合索引范围 0..80。必须同时返回坐标和组合索引：prior 预测坐标，评测/检索可使用组合 ID。

E_r 读取 x_r、动作和 detached/straight-through z 特征，输出 mu_r/logvar_r。训练采样：

    r = mu_r + exp(0.5 * logvar_r) * epsilon

whole-vector dropout 在样本维一次性丢掉完整 r，不能独立丢每一维，否则 residual 仍可通过剩余维稳定绕过 z。

### 7.3 hybrid_tokenizer.py

forward 的返回值至少包括：

- z_continuous、z_quantized、z_index；
- r_mu、r_logvar、r_sample；
- coarse outcome/contact logits；
- precise outcome/force prediction；
- 每类 loss 的独立统计。

不要在模块内部直接加权求一个总 loss。训练脚本先按有效元素和经验尺度归一化，再由配置组合，日志中保留每项原始值，避免辅助损失悄悄压过 action loss。

### 7.4 boundary.py 与 termination.py

必须是两个不同网络：

- OfflineBoundaryPosterior：允许双向 temporal attention，只生成训练分段；
- CausalTerminationHead：只看过去，输出离散时间 hazard。

二者最多共享逐帧、不跨时间的 stem。单测应把 t 之后的输入全部随机改写，确认 t 时刻 termination 完全不变。

timeout、故障、安全中断是 censored；只有 event/manual 和确认自然结束的 episode_end 才是正常 termination 正例。

### 7.5 subgoal.py

每个样本输出 K_g=3 个 proposal。每个 proposal 包含：

    geometry, visual_latent(128), contact_logits(7),
    proposal_score, feasibility

融合必须显式使用 modality mask。三种目标模态全缺时 feasibility 强制为 0。部署排序不能只看 proposal_score，至少要先过 safety/feasibility gate，再计算该 g 下的 skill prior。

### 7.6 skill_prior.py

因果顺序固定为：

    context0 = E_ctx(current/history, language, skill_history)
    proposals = G(context0)
    context  = E_skillctx(context0, chosen_g)
    z_logits = p(z | context)       # 4 个三分类，逐坐标自回归
    r_dist   = p(r | context,z,g)   # diagonal Gaussian

训练时 z_teacher 与 q(r|segment) 必须 stop-gradient。部署时不创建 trajectory encoder，更不能在代码中留一个可选 posterior fallback。

## 8. π0/π0.5 的精确接入

### 8.1 为什么接 embed_suffix

OpenPI 的视觉/语言 prefix 会先进入大 PaliGemma/SigLIP；动作 expert 的 suffix 读取 prefix，并预测 flow velocity。CASK 的 z/r/g 是“如何执行当前短时目标”的条件，放入 suffix 能做到：

- prefix 不反向读取未来/动作条件；
- action tokens 能读取完整 skill condition；
- 不改变 tokenizer vocabulary；
- 基座 checkpoint 的视觉语言参数路径不变。

### 8.2 两个 token

    mode_token     = ModeProjector(z_coords)
    instance_token = InstanceProjector([r, g, observed_event, validity])

第一版不把 z 直接转成文本 token。学习到的连续 suffix token 不占 PaliGemma vocabulary，也不会把 81 个组合误当作自然语言语义。

π0 suffix 顺序：

    state | mode | instance | action_1 ... action_H

block mask：

    True, True, False, True, False ... False

π0.5 suffix 顺序：

    mode | instance | action_1 ... action_H

block mask：

    True, False, True, False ... False

mode/instance 同属一个双向块；action chunk 属于下一个双向块，因此动作可读两个 skill token，而 skill token 不读带噪 action。

### 8.3 不应直接覆盖 Pi0.compute_loss

建议保留标准 compute_loss/sample_actions 以复现原模型，新增：

    compute_cask_loss(rng, observation, actions, condition, action_mask)
    sample_cask_actions(rng, observation, condition, action_mask, num_steps)

Stage 4 自定义训练器调用新 API；direct π baseline 仍调用原 API。这样一次 checkout 即可做严格对照，也防止无意把 action mask 或 skill token 带进基线。

### 8.4 π0.5 的 adaRMS

首版只插 suffix tokens，不改 adaRMS。只有 token-only 通过形状、加载和基线等价测试后，才试：

    adarms_cond = time_condition + gate * SkillAdaProjector(skill_condition)

gate 初始为 0。把 skill 同时注入 token 和每层 adapter 会增加归因混淆，所以必须分别报告 token-only、adapter-only、token+adapter，以及参数量匹配但输入为零/随机的 adapter control。

### 8.5 32D 动作槽

定义不可变 ActionSlotMap，例如 native 第 i 维映射到 π slot j。pack、mask、unpack 使用同一份配置和哈希。

训练时依次 mask：

    actions -> noise -> x_t -> u_t -> predicted_v -> loss

采样时依次 mask：

    initial_noise -> every x_t input -> predicted_v -> every Euler update -> final_actions

只在最终输出 mask 不够，因为无效槽已经通过 action_in_proj 和 self-attention 影响有效槽。

## 9. 多阶段训练与冻结

### Stage 0：先证明数据可做

运行 audit_dataset.py，输出：

- episode/session/object 分组后的 split manifest 和 SHA-256；
- task、FPS、native action dim、传感器覆盖；
- contact/event 候选数量；
- 跨任务兼容 segment 候选对；
- 原生动作到 32D slot 的可逆性。

若 G0 没有足够的跨任务复用阶段，应先改任务/采集，不训练“共享技能”。

### Stage 1：不加载 π

固定 8/16/32 帧或等秒窗，冻结视觉特征，训练 HybridTokenizer + 4 层 conditional flow decoder。统一 decoder、训练步数、latent bitrate 和搜索预算比较：

- discrete-only；
- continuous-only；
- z+r。

### Stage 2：替换窗口来源

先冻结 tokenizer，训练 boundary；生成带 reason 的 segment manifest；再用新 segment 重训 tokenizer。hard NMS/merge 不反传。训练 causal termination 并单独评估。

### Stage 3：训练完整因果选择链

posterior 只提供 detached target。subgoal 从 teacher target 按 curriculum 切到 predicted；prior 做坐标 CE、Gaussian KL 与 prefix residual-to-go。验证必须同时报告 oracle 条件与 predicted chain 的差距。

### Stage 4：接 π0.5

推荐三段：

1. 冻结 tokenizer、prior、PaliGemma/SigLIP；posterior condition warm-up 新 skill projector/action expert；
2. predicted g/z/r 比例 0→50%→100%；categorical z 不接 flow 梯度，连续 r 可接；
3. 只在需要时打开 action expert LoRA/小学习率；保留 direct π0.5 参数匹配基线。

### Stage 5：few-shot

对 held-out 任务使用相同的 1/5/10/20/50 demo 清单，比较：

- direct π；
- subgoal-only；
- skill-only；
- subgoal+skill。

tokenizer 默认冻结。分别尝试 prior-only、prior+residual adapter、LoRA、full FT，报告实际可训练参数量。

## 10. 部署状态机

    PROPOSE
      生成 3 个 g proposal
        ↓
    FEASIBILITY_CHECK
      过滤低可达/不安全候选
        ↓
    EXECUTE
      选 z，预测 r，π0.5 生成 chunk，只执行前 h_exec
        ↓
    VERIFY
      更新观测、r 和 hazard
      ├─ 未结束：保持 z/g，回到 EXECUTE
      ├─ 正常结束：清理当前 skill，回到 PROPOSE
      ├─ timeout：续段保持 z，或触发安全重规划
      └─ 两次无进展/安全触发/候选耗尽：SAFE_STOP

部署日志必须记录每一步的 condition_source。主结果只接受 predicted g/z/r + causal termination；任何 posterior/oracle 调用都应使 rollout 标记为 oracle，而不是静默继续。

## 11. 必须先写的测试

1. Mask separation：任意改无效 feature/sensor/action 槽不改变有效输出；
2. ODE mask：每一步无效动作槽严格为 0；
3. Future leakage：改未来 segment 不改变 causal prior/termination；
4. Token block mask：π0 与 π0.5 的 suffix 长度和 block 编号精确匹配；
5. Baseline equivalence：关闭 skill path 时，固定 RNG 下与原 Pi0 输出一致；
6. Checkpoint load：基座参数无 missing shape，新参数只来自 CASK；
7. Posterior ban：部署 policy 的调用图不包含 segment encoder/offline boundary；
8. Skill swap：只换 z，在目标 context 重算 r；
9. Censoring：timeout 不产生正常 termination 正标签；
10. Split leakage：同 session/object/近重复 episode 不跨 split。

## 12. 当前可用默认值与必须外部确认的参数

可作为代码默认、但仍需消融：

- skill rate：10 Hz；
- segment encoder：4 层、width 256、8 heads；
- FSQ：levels=(3,3,3,3)；
- residual_dim=16；
- subgoal_dim=256、K_g=3；
- residual whole-vector dropout=0.2；
- π action_dim=32；
- Stage 4 主基座：pi05=True。

必须从数据/机器人/算力确认，不能在配置中假装已知：

- native action schema 与 slot map；
- control FPS、π horizon、h_exec；
- tau_min/tau_max；
- camera key 与同步误差；
- object pose/关键点来源；
- contact 真值/代理/置信度；
- D_geo；
- GPU 数量/显存与 rollout 预算。

## 13. 环境与运行约束

本地当前是 Windows，未发现可用 nvidia-smi、JAX、Flax 或 Torch 环境；OpenPI 官方 README 又明确只测试/支持 Ubuntu 22.04。因此本地可以完成源码、静态编译和文档审计，但不能据此宣称模型运行通过。

推荐在 Ubuntu 22.04 GPU 节点使用 ../openpi 的 Python 3.11/uv 环境。官方给出的单卡显存估计为：

- inference：大于 8 GB；
- LoRA fine-tuning：大于 22.5 GB；
- full fine-tuning：大于 70 GB。

首次远端验证顺序应为：导入检查 → dummy shape test → 单 batch forward/backward → 100 step overfit → 小数据 rollout，而不是直接提交完整训练。

## 14. 官方来源

- OpenPI：https://github.com/Physical-Intelligence/openpi （Apache-2.0）
- π0：https://www.pi.website/blog/pi0
- π0 paper：https://arxiv.org/abs/2410.24164
- FAST：https://www.pi.website/research/fast
- FAST paper：https://arxiv.org/abs/2501.09747
- π0.5：https://www.pi.website/blog/pi05
- π0.5 paper：https://arxiv.org/abs/2504.16054
- Knowledge insulation：https://www.pi.website/research/knowledge_insulation

## 15. 实施状态

- [x] 审计 CASK v0.2 设计与信息泄漏边界；
- [x] 核对本地 OpenPI commit、π0/π0.5 suffix、flow loss、ODE 和动作维；
- [x] 核对 π0/FAST/π0.5 官方论文与实现支持范围；
- [x] 冻结 JAX/Flax 和独立 overlay 方案；
- [x] 建立 config/types/mask/FSQ/π adapter 的首版代码；
- [x] 建立 segment encoder、hybrid tokenizer、MVP flow、boundary、termination、subgoal、skill prior 和状态机骨架；
- [x] 完成全量静态编译与不依赖 JAX 的配置/布局/状态机契约测试；
- [x] 提供 CASK core 和 dummy π0.5 两个 JAX smoke 脚本；
- [x] 在隔离 CPU JAX/Flax + 当前 OpenPI 源码上执行 core/π smoke 与 pytest；
- [ ] 获得数据路径后执行 M0/G0；
- [ ] 在 Ubuntu GPU 节点完成 JAX runtime 测试。

### 2026-08-14 — 检查点 D：首版代码落地与验证证据

已创建的实际模型代码：

| 文件 | 已实现内容 |
|---|---|
| cask/config.py | 结构默认值、M0 必填项、slot/timing 校验 |
| cask/types.py | 训练 segment、部署 context、skill condition 的类型隔离 |
| cask/masking.py | 32D pack/unpack、逐槽 mask、有效元素 flow MSE |
| cask/models/fsq.py | 4×3 FSQ、straight-through、index/code 互转 |
| cask/models/segment_encoder.py | masked temporal Transformer + attention pooling |
| cask/models/hybrid_tokenizer.py | 双分支 posterior q(z,r|segment) |
| cask/models/mvp_flow_decoder.py | Stage 1 conditional flow decoder |
| cask/models/boundary.py | 离线双向 boundary posterior |
| cask/models/termination.py | 独立 causal hazard head |
| cask/models/subgoal.py | 3 候选、多模态 mask-aware subgoal |
| cask/models/skill_prior.py | 逐坐标 FSQ prior + Gaussian residual prior |
| cask/models/pi0_skill_adapter.py | opt-in CaskPi0/CaskPi05、skill tokens、masked loss/ODE |
| cask/losses/core.py | Gaussian KL、capacity KL、FSQ CE、删失 hazard NLL |
| cask/policies/cask_policy.py | 可独立测试的部署状态机 |

初轮验证结果：

- Python compileall 对 cask/、scripts/、tests/ 全部通过；
- 默认配置校验、81 个 FSQ 组合、M0 缺失项阻断通过；
- π0 与 π0.5 suffix block mask 纯契约测试通过；
- PROPOSE→CHECK→EXECUTE→VERIFY→PROPOSE 状态流纯契约测试通过；
- scripts/audit_environment.py 已实际运行，确认本地 OpenPI HEAD 仍为固定 commit；
- 当前 Windows 运行环境无 JAX/Flax/Torch/OpenPI/pytest，也没有可见 nvidia-smi；
- 后来出现的 ../openpi/.venv 是 Python 3.11.15，但同样尚未安装上述依赖；
- 该结论随后由检查点 E 的隔离 CPU runtime 验证补强。

### 2026-08-14 — 检查点 E：隔离 JAX/OpenPI runtime 验证

为避免修改用户已有 Conda/OpenPI 环境，依赖临时安装到工作区内的 .cask_test_deps，测试完成后已经删除。测试宿主是 Windows Python 3.12；JAX/Flax 版本与固定 OpenPI 对齐：

- JAX/JAXlib 0.5.3（Windows CPU wheel）；
- Flax 0.10.2；
- 当前 ../openpi 源码，HEAD 仍为 15a9616；
- OpenPI 相关工作树变化主要是另一任务的中文架构注释及无语义局部清理，本任务未覆盖它们。

实际结果：

1. scripts/smoke_cask_core.py：PASS；
2. scripts/smoke_cask_pi.py：PASS；
3. 全套 pytest：15 passed；
4. π0 与 π0.5 suffix mask 均通过；
5. CaskPi0 继承的标准 sample_actions 与原 Pi0 固定 RNG 输出等价；
6. CASK masked flow loss 和两步 ODE 通过，无效动作槽严格为 0；
7. causal termination 的未来扰动不改变过去 hazard；
8. FSQ 81 个 index/code 全量 round-trip；
9. 临时依赖、pytest cache 和 pycache 已清理。

测试出现 748 条 warning，均来自固定 JAX/Flax 内部对未来 API 的 deprecation 提示；没有 assertion、shape 或数值失败。它们提醒未来升级 JAX/Flax 时要重新跑兼容性审计，但不否定当前固定版本测试。

证据边界：以上是 CPU dummy/small-model 的接口和不变量验证，证明代码路径可执行；它不等于真实 checkpoint、GPU mixed precision、FSDP、真实数据 backward 或机器人 rollout 已通过。后四项仍需 Ubuntu GPU 节点和 M0 数据输入。

## 16. 在支持环境中的运行顺序

在 Ubuntu 22.04 GPU 节点：

    cd /path/to/mrc_research
    uv sync --group dev
    uv run python scripts/audit_environment.py
    uv run python scripts/smoke_cask_core.py
    uv run python scripts/smoke_cask_pi.py
    uv run pytest -q

审计脚本只有在 Python 3.11、Linux、JAX、Flax、OpenPI 和 NVIDIA GPU 均存在时才输出 runtime_ready=true。

两个 smoke 的职责不同：

1. smoke_cask_core.py 用极小宽度验证所有独立 CASK 模块的 init/apply 和 shape；
2. smoke_cask_pi.py 使用 OpenPI dummy PaliGemma/action expert，验证 skill suffix、masked flow loss、两步 ODE 和无效动作槽恒为 0。

正式训练前还要增加：

    单 batch forward/backward
      → 100 step 小样本过拟合
      → 固定 seed checkpoint save/restore
      → 原 π baseline 等价
      → 小规模 prior-only rollout

任何一步失败都应停在该层修复，不能跳到完整数据训练。

## 17. 最小 Stage 4 调用关系

训练器不调用原 compute_loss，而是显式传入 condition 和 action_mask：

    posterior = tokenizer(segment_batch)              # 训练 warm-up
    subgoals  = subgoal_head(causal_context)
    prior     = skill_prior(causal_context, subgoals)  # predicted curriculum
    condition = choose_condition(posterior, prior, schedule)
    per_step_loss, time_valid = pi.compute_cask_loss(
        rng, observation, actions, condition, action_mask
    )
    flow_loss = masked_mean(per_step_loss, time_valid)

主验证/部署路径必须删除第一行 posterior 调用：

    g = select_feasible(subgoal_head(causal_context))
    z, r = skill_prior(causal_context, g)
    condition = SkillCondition(z, r, g, observed_event, validity)
    actions = pi.sample_cask_actions(
        rng, observation, condition, action_mask
    )

这两段调用关系是训练/部署防泄漏的最终代码边界。

### 2026-08-14 — 检查点 F：交付审计

按原目标逐项复核：

| 原目标要求 | 当前证据 | 结论 |
|---|---|---|
| 基于目前模型设计 | 逐节读取 model_design.md v0.2，并把三层职责、z+r、subgoal、prior、termination、π executor 映射到实际模块 | 完成 |
| 基于 π 系列 | 官方核验 π0、π0-FAST、π0.5；固定 OpenPI commit；给出选型、差异和接入点 | 完成 |
| 构建模型代码 | 独立 cask overlay、配置、核心模型、π adapter、loss、状态机、smoke 与测试均已创建 | 完成首版实现骨架 |
| 详细解析和解释 | 本文覆盖数据流、张量、mask、梯度、训练阶段、部署、测试、风险和来源 | 完成 |
| 实时记录本地 Markdown | 检查点 A–F 连续记录调查、决策、实现和验证 | 完成 |

当前共有 29 个 Python 源码/测试/脚本文件通过 AST/静态编译审计；关键 20 个交付文件存在性检查通过；Markdown UTF-8、代码围栏与必需章节检查通过；部署 policy 和 π adapter 均不导入 TrainingSegmentBatch、HybridSkillTokenizer 或 OfflineBoundaryPosterior。

尚未完成的 M0 数据审计、真实 checkpoint、Ubuntu GPU backward 和机器人 rollout 不是文档/首版模型骨架的隐藏完成项，已经作为下一工程阶段显式保留。开始这些工作前必须由真实数据路径、机器人动作 schema 和算力环境提供证据，不能用当前 dummy 测试替代。
