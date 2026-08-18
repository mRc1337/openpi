# π0.5 在 LIBERO 上的微调复现（zxw 本地路径版）

本文是在当前仓库版本上，完整复现官方 **π0.5 + LIBERO 全量微调** 的操作手册。训练超参数与官方
`pi05_libero` 配置保持一致，只把数据集、基础权重、缓存、归一化统计、训练 checkpoint、日志和评测视频
统一保存到 zxw 的个人目录。

本文对应的代码版本：

```text
仓库：/home/pai/zxw/openpi
分支：main
提交：45f9b64
训练配置：pi05_libero_zxw
说明：下文第 16 节列出的复现修改当前位于该提交之上的工作树中
```

官方发布 checkpoint 的参考结果如下（每个 suite 每个任务 50 次 rollout）：

| 模型 | Spatial | Object | Goal | LIBERO-10 | 平均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| π0.5 @ 30k | 98.8 | 98.2 | 98.0 | 92.4 | 96.85 |

> 这里的“复现”指使用同一公开数据、模型结构、训练超参数和评测设置重新训练并评测。由于 GPU、JAX、
> CUDA 算子及数据读取顺序等因素，单次训练结果不保证逐位相同，也不应要求恰好得到 96.85。

## 1. 本仓库为 zxw 做过的路径调整

新增的 `pi05_libero_zxw` 与官方 `pi05_libero` 保持以下关键参数一致：

| 项目 | 值 |
| --- | --- |
| 模型 | `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)` |
| 数据集 | `physical-intelligence/libero` |
| batch size | 256（全局 batch） |
| 训练步数 | 30,000 |
| warmup | 10,000 |
| 学习率 | `5e-5`，官方配置中 decay 前后均为 `5e-5` |
| 梯度裁剪 | 1.0 |
| EMA | 0.999 |
| 初始权重 | `gs://openpi-assets/checkpoints/pi05_base/params` |
| action delta 变换 | 关闭（LIBERO action 已经是 delta） |

只有以下落盘路径被改为 zxw 的目录：

```text
assets_base_dir     = /home/pai/zxw/openpi_data/pi05_libero/assets
checkpoint_base_dir = /home/pai/zxw/openpi_data/pi05_libero/checkpoints
```

另外，`scripts/train.py` 已支持读取 `JAX_COMPILATION_CACHE_DIR`，不再强制把编译缓存写到
`~/.cache/jax`。

## 2. 硬件与磁盘要求

官方 README 给出的单卡显存下限是：

| 模式 | 显存要求 |
| --- | ---: |
| 推理 | 大于 8 GB |
| LoRA 微调 | 大于 22.5 GB |
| 全量微调（本复现） | 大于 70 GB |

本配置是全量微调，并不是 LoRA。单卡建议使用 80 GB A100/H100；多卡时可用 `fsdp_devices` 分片。
当前仓库训练器支持单机多卡，不支持多节点。

checkpoint 内含参数、优化器和 EMA 状态，且每 5,000 步保留一个周期 checkpoint。建议开始前给
`/home/pai/zxw/openpi_data/pi05_libero` 预留至少数百 GB，并在训练过程中用 `du` 持续观察；实际占用取决于
checkpoint 序列化和最终保留数量。

开始前检查：

```bash
cd /home/pai/zxw/openpi
source examples/libero/zxw_env.sh

nvidia-smi
df -h /home/pai/zxw
du -sh "$PI05_LIBERO_ROOT" 2>/dev/null || true
```

如果 `nvidia-smi` 不能正常列出 GPU，先修复宿主机 NVIDIA 驱动或作业调度分配；训练不能在 CPU 上实际完成。

## 3. 大文件目录规划

每次打开新终端后，第一件事都是执行：

```bash
cd /home/pai/zxw/openpi
source examples/libero/zxw_env.sh
```

该脚本会创建并导出以下目录：

```text
/home/pai/zxw/openpi_data/pi05_libero/
├── assets/                       # norm_stats.json
├── checkpoints/                  # 训练 checkpoint
├── cache/
│   ├── openpi/                   # GCS 基础权重及 PaliGemma tokenizer
│   ├── huggingface/
│   │   └── lerobot/              # physical-intelligence/libero
│   ├── uv/                       # Python wheel/source 缓存
│   ├── jax/                      # JAX 编译缓存
│   ├── cuda/                     # CUDA kernel 缓存
│   ├── triton/                   # Triton 缓存
│   └── torch_extensions/         # PyTorch 扩展缓存
├── logs/                         # 训练和评测文本日志
├── tmp/                          # 临时大文件
├── python/                       # uv 自动下载的 Python 解释器
├── venvs/                        # 独立的 LIBERO Python 3.8 环境
├── videos/                       # rollout 视频
└── wandb/                        # W&B 本地运行记录
```

确认关键环境变量没有落到 root 或系统目录：

```bash
env | grep -E '^(OPENPI_DATA_HOME|HF_HOME|HF_LEROBOT_HOME|UV_CACHE_DIR|UV_PYTHON_INSTALL_DIR|JAX_COMPILATION_CACHE_DIR|WANDB_DIR|TMPDIR)='
```

预期每一行都位于 `/home/pai/zxw/openpi_data/pi05_libero` 下。

## 4. 初始化代码和 Python 3.11 训练环境

LIBERO 评测代码位于 git submodule 中。submodule 会拉到当前 zxw 仓库的 `third_party/`，不会写入系统目录。

```bash
cd /home/pai/zxw/openpi
source examples/libero/zxw_env.sh

git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

说明：

- 项目训练环境默认位于 `/home/pai/zxw/openpi/.venv`；它仍在 zxw 的个人目录内。
- uv 的下载缓存由 `UV_CACHE_DIR` 指向个人数据目录。
- `zxw_env.sh` 同时设置了当前 uv 推荐的 `UV_DEFAULT_INDEX` 和兼容变量 `UV_INDEX_URL`：
  `https://mirrors.aliyun.com/pypi/simple/`。2026-08-18 已确认该镜像返回 HTTP 200。
- `GIT_LFS_SKIP_SMUDGE=1` 是本项目安装 LeRobot 依赖时的官方要求。
- `--frozen` 保证严格使用当前 `uv.lock`，避免无意升级依赖。

需要注意：当前 `uv.lock` 中 registry 是 `https://pypi.org/simple`，并记录了 2,245 个
`files.pythonhosted.org` 分发文件 URL。`uv sync --frozen` 为保证锁文件可复现，会继续从这些固定 URL 下载部分
wheel；阿里云镜像会用于索引访问、未固定的构建依赖及其他 pip/uv 操作，但不会重写 frozen lock 中的 URL。
实测当前 uv 缓存同时出现阿里云和 `files.pythonhosted.org` 来源，属于预期行为。

不要为了全部切到镜像而在正式复现中去掉 `--frozen`，否则 uv 可能重写 `uv.lock`，依赖来源和解析结果就不再与
仓库当前版本严格一致。当前同步可安全中断并原样重跑，`$UV_CACHE_DIR` 中已经完成的 wheel 和解包结果会复用。

安装后验证版本和设备：

```bash
uv run python -c 'import jax, torch; print("JAX", jax.__version__, jax.devices()); print("Torch", torch.__version__, torch.cuda.is_available())'
uv run python -c 'from openpi.training import config; c=config.get_config("pi05_libero_zxw"); print(c)'
```

第二条命令显示的 `assets_base_dir` 和 `checkpoint_base_dir` 必须是第 1 节中的绝对路径。

## 5. 下载 π0.5 基础权重

### 5.1 当前节点使用代理下载（已验证）

当前节点的代理可供 `curl` 使用，但 `gsutil` 会收到 `407 Proxy Authentication Required`。因此新增了
`scripts/download_pi05_base_zxw.py`：它通过当前 `HTTP_PROXY` / `HTTPS_PROXY` 调用 `curl`，6 路并发下载，
每个对象支持 `.part` 断点续传，逐文件检查官方字节数，并在全部校验通过后才把 `params.partial` 原子改名为
`params`。

```bash
cd /home/pai/zxw/openpi
source examples/libero/zxw_env.sh

python scripts/download_pi05_base_zxw.py --workers 6
```

下载中断后直接重跑同一命令，完整对象会跳过，未完成对象从 `.part` 当前字节继续。成功输出应包含：

```text
Checkpoint verified: 20 objects, 12441721931 bytes
Published checkpoint: /home/pai/zxw/openpi_data/pi05_libero/cache/openpi/openpi-assets/checkpoints/pi05_base/params
```

正式权重路径为：

```text
/home/pai/zxw/openpi_data/pi05_libero/cache/openpi/openpi-assets/checkpoints/pi05_base/params
```

独立复核：

```bash
WEIGHT_DIR="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params"
test -d "$WEIGHT_DIR"
du -sh "$WEIGHT_DIR"
find "$WEIGHT_DIR" -type f | wc -l
find "${WEIGHT_DIR%/params}" -type f \( -name '*.part' -o -name '*.gstmp' \) -print
python scripts/download_pi05_base_zxw.py --workers 6
```

预期约 `12G`、文件数 `20`、临时文件查询无输出，最后一条命令报告 `Checkpoint is already complete`。

### 5.2 2026-08-18 实际下载记录

本次已经完成下载，不需要再次下载：

| 项目 | 实际结果 |
| --- | --- |
| GCS 来源 | `gs://openpi-assets/checkpoints/pi05_base/params` |
| GCS 远端大小 | 11.59 GiB |
| 正式本地路径 | `/home/pai/zxw/openpi_data/pi05_libero/cache/openpi/openpi-assets/checkpoints/pi05_base/params` |
| 对象数 | 20 |
| 精确总字节数 | 12,441,721,931 |
| `du -sh` | 12G |
| 残留 `.part` / `.gstmp` | 无 |
| 最大对象 | 3,077,149,148 字节，已校验 |

切换下载器时清理了 9 个仅供中断 `gsutil` 使用、不能由 curl 正确续传的稀疏 `.gstmp` 文件；代理测速生成的
`/tmp/pi05_proxy_test.bin`（100 MiB）也已在测速后删除。正式权重未被删除或覆盖。

### 5.3 仍会按需下载的 tokenizer

模型在数据预处理时还会自动拉取：

```text
gs://big_vision/paligemma_tokenizer.model
```

它同样受 `OPENPI_DATA_HOME` 控制，最终位于个人目录的：

```text
$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model
```

## 6. 下载 LIBERO 数据并计算归一化统计

官方复现直接使用已经转换好的 LeRobot 数据集 `physical-intelligence/libero`，不需要再下载 RLDS 原始数据或
运行 `convert_libero_data_to_lerobot.py`。第一次计算统计量时，LeRobot 会自动把数据下载到
`HF_LEROBOT_HOME`。

### 6.1 当前代理下的加速设置

`zxw_env.sh` 已启用 Hugging Face Xet 高性能模式，并继承当前终端已有的 `HTTP_PROXY` / `HTTPS_PROXY`：

```text
HF_ENDPOINT=https://hf-mirror.com
HF_XET_HIGH_PERFORMANCE=1
HF_XET_NUM_CONCURRENT_RANGE_GETS=16
HF_HUB_DOWNLOAD_TIMEOUT=600
HF_HUB_ETAG_TIMEOUT=600
HF_XET_CACHE=/home/pai/zxw/openpi_data/pi05_libero/cache/huggingface/xet
```

2026-08-18 对 LIBERO parquet 做过 10 MiB 代理测速，单连接约 2.2 MiB/s；Xet 会使用并发 range GET。当前
`physical-intelligence/libero` 仓库至少约 23 GB，因此即使代理正常也需要一定时间。

当前节点直接访问 `huggingface.co` 会报 `Network is unreachable`，且用户终端不一定带有工具使用的代理变量。
因此 `zxw_env.sh` 默认设置国内 endpoint `https://hf-mirror.com`。已在完全取消代理的情况下验证 LIBERO refs
API 返回 HTTP 200，并验证 parquet range 下载返回 HTTP 206。这里不再设置已被 Transformers 弃用的
`TRANSFORMERS_CACHE`，统一由 `HF_HOME` 控制缓存，避免 FutureWarning。

必须在启动 `compute_norm_stats.py` 的同一个终端重新 source，使新变量进入当前 shell：

```bash
cd /home/pai/zxw/openpi
source examples/libero/zxw_env.sh

env | grep -E '^(HF_ENDPOINT|HF_XET_HIGH_PERFORMANCE|HF_XET_NUM_CONCURRENT_RANGE_GETS|HF_HUB_DOWNLOAD_TIMEOUT|HF_HUB_ETAG_TIMEOUT|HF_XET_CACHE)='
uv run python -c 'import hf_xet, huggingface_hub; print("hf_xet enabled; hub", huggingface_hub.__version__)'

uv run scripts/compute_norm_stats.py --config-name pi05_libero_zxw \
  2>&1 | tee "$PI05_LIBERO_ROOT/logs/compute_norm_stats.log"
```

如果之前的命令仍停留在 `uv sync --frozen`，说明它正在下载/解包 JAX、Torch、CUDA 等 Python 依赖，尚未开始
LIBERO 数据下载。uv 已经使用当前代理和 `UV_INDEX_URL`，可让它继续完成；若希望紧接着启用上述新变量，可
按 `Ctrl-C` 停止一次，重新 source `zxw_env.sh` 后原样重跑。uv 会复用 `$UV_CACHE_DIR`，不会从零下载。

如果已经进入 LIBERO 下载阶段，也可安全停止后重新 source 并重跑；Hugging Face 会复用已完成文件及缓存。

若下载若干文件后出现 `Read timed out. (read timeout=10)` 和 `LocalEntryNotFoundError`，这是 Hugging Face 的
HEAD/etag 元数据请求仍使用独立的 10 秒默认超时，不代表已下载文件损坏。当前环境已设置
`HF_HUB_ETAG_TIMEOUT=600`；重新 source 后原样重跑即可续传，不要删除 LeRobot/Hugging Face 缓存。

数据集提示 `Revision v2.1 ... not found, using version v2.0` 以及全局 stats 警告是当前官方公开 LIBERO 数据的
正常兼容提示。此仓库锁定的 LeRobot 对 v2.0 向后兼容，为保持 openpi 官方复现，不要运行提示中的 v2.0 到
v2.1 转换脚本。

### 6.2 计算全量归一化统计

不要为正式复现传 `--max-frames`；该参数只适合快速调试，会让统计量不是全数据统计。

成功后应生成：

```text
/home/pai/zxw/openpi_data/pi05_libero/assets/
└── pi05_libero_zxw/
    └── physical-intelligence/
        └── libero/
            └── norm_stats.json
```

检查文件和数据位置：

```bash
NORM_STATS="$PI05_LIBERO_ROOT/assets/pi05_libero_zxw/physical-intelligence/libero/norm_stats.json"
test -s "$NORM_STATS"
python -m json.tool "$NORM_STATS" >/dev/null

du -sh "$HF_LEROBOT_HOME/physical-intelligence/libero" 2>/dev/null || du -sh "$HF_HOME"
du -sh "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
```

如果下载 Hugging Face 数据需要代理或镜像，应在 `source examples/libero/zxw_env.sh` 后再设置站点所需变量；
不要改变 `HF_HOME` 和 `HF_LEROBOT_HOME`。

## 7. 启动正式 30k 训练

### 7.1 W&B 设置

在线记录：

```bash
uv run wandb login
```

无外网时可离线记录，数据仍保存在 zxw 目录：

```bash
export WANDB_MODE=offline
```

### 7.2 单卡 A800 全量微调完整指令

以下命令用于单张 80 GB A800 上的官方配置全量微调，不是 LoRA。必须等待 norm stats 命令正常结束后才能启动
训练。

#### 7.2.1 初始化正确环境

```bash
cd /home/pai/zxw/openpi

unset VIRTUAL_ENV
unset TRANSFORMERS_CACHE
source .venv/bin/activate
source examples/libero/zxw_env.sh

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export EXP_NAME="pi05_libero_full_30k_seed42_a800x1"

set -o pipefail
```

检查当前激活的确实是 openpi 环境，并确认 JAX 只看到一张 A800：

```bash
echo "$VIRTUAL_ENV"
nvidia-smi -L
uv run python -c 'import jax; devices=jax.devices(); print(devices); assert len(devices) == 1'
```

`VIRTUAL_ENV` 应为 `/home/pai/zxw/openpi/.venv`。如果显示其他项目路径，不要继续训练。

#### 7.2.2 检查基础权重

```bash
WEIGHT_DIR="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params"

test -d "$WEIGHT_DIR"
du -sh "$WEIGHT_DIR"
python scripts/download_pi05_base_zxw.py --workers 6
```

最后一条应报告 `Checkpoint is already complete`。当前已验证权重为 20 个对象、12,441,721,931 字节。

#### 7.2.3 下载数据并生成 norm stats

```bash
echo "$HF_ENDPOINT"
echo "$HF_HUB_ETAG_TIMEOUT"

uv run scripts/compute_norm_stats.py \
  --config-name pi05_libero_zxw \
  2>&1 | tee -a "$PI05_LIBERO_ROOT/logs/compute_norm_stats.log"
```

等待命令正常退出后检查：

```bash
NORM_STATS="$PI05_LIBERO_ROOT/assets/pi05_libero_zxw/physical-intelligence/libero/norm_stats.json"

test -s "$NORM_STATS"
python -m json.tool "$NORM_STATS" >/dev/null
ls -lh "$NORM_STATS"
```

如果 `test -s` 返回非零，说明统计文件尚未生成，不能启动训练。

#### 7.2.4 配置 W&B

没有 W&B 登录或外网不稳定时使用离线模式：

```bash
export WANDB_MODE=offline
```

需要在线记录时改为：

```bash
unset WANDB_MODE
uv run wandb login
```

#### 7.2.5 启动 30k 全量微调

```bash
uv run scripts/train.py pi05_libero_zxw \
  --exp-name "$EXP_NAME" \
  --fsdp-devices 1 \
  2>&1 | tee "$PI05_LIBERO_ROOT/logs/${EXP_NAME}.log"
```

该命令使用官方 `pi05_libero` 等价参数：全量微调、全局 batch size 256、30,000 步、seed 42。checkpoint
写入：

```text
/home/pai/zxw/openpi_data/pi05_libero/checkpoints/pi05_libero_zxw/pi05_libero_full_30k_seed42_a800x1
```

首次运行不要添加 `--overwrite`。如果同名目录已经存在，程序会主动停止，防止误删 checkpoint。

#### 7.2.6 中断后续训

重新执行 7.2.1 的环境初始化，并使用完全相同的实验名：

```bash
export WANDB_MODE=offline
export EXP_NAME="pi05_libero_full_30k_seed42_a800x1"

uv run scripts/train.py pi05_libero_zxw \
  --exp-name "$EXP_NAME" \
  --fsdp-devices 1 \
  --resume \
  2>&1 | tee -a "$PI05_LIBERO_ROOT/logs/${EXP_NAME}.log"
```

### 7.3 当前节点 4×A800 的 FSDP 训练命令

用户终端已确认当前节点有 4 张 A800。工具执行环境与宿主机 GPU namespace 隔离，因此本次工具内的
`nvidia-smi` 不可见 GPU；训练前仍应在实际训练终端运行以下两条命令确认 JAX 也看到 4 卡：

```bash
nvidia-smi -L
uv run python -c 'import jax; print(jax.devices()); assert len(jax.devices()) == 4'
```

完成第 6 节全量 norm stats 后，使用以下正式命令把模型分片到 4 张 A800：

```bash
cd /home/pai/zxw/openpi
source examples/libero/zxw_env.sh

export EXP_NAME="pi05_libero_full_30k_seed42_a800x4"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

set -o pipefail
uv run scripts/train.py pi05_libero_zxw \
  --exp-name "$EXP_NAME" \
  --fsdp-devices 4 \
  2>&1 | tee "$PI05_LIBERO_ROOT/logs/${EXP_NAME}.log"
```

注意：

- `batch_size=256` 是全局 batch，必须能被可见的 JAX device 数整除。
- `fsdp_devices` 必须能合理划分可见设备。例如总共 8 卡且 `fsdp_devices=4` 时，会形成两个数据并行组。
- 为保持官方复现，不要因为显存不足擅自降低 batch、训练步数或切换 LoRA；应优先增加显存或卡数。

### 7.4 中断后续训

必须使用完全相同的 `EXP_NAME` 和卡配置，并把新训练命令改为 `--resume`：

```bash
uv run scripts/train.py pi05_libero_zxw \
  --exp-name "$EXP_NAME" \
  --resume \
  2>&1 | tee -a "$PI05_LIBERO_ROOT/logs/${EXP_NAME}.log"
```

多卡实验续训时还要保留相同的 `--fsdp-devices N`。

只有在明确要删除同名实验并从头训练时才使用 `--overwrite`。该选项会清空对应实验 checkpoint 目录，不能和
`--resume` 同时使用。

## 8. 训练期间与训练完成后的检查

checkpoint 根目录为：

```text
$PI05_LIBERO_ROOT/checkpoints/pi05_libero_zxw/$EXP_NAME/
```

查看保存步、空间和错误日志：

```bash
CKPT_ROOT="$PI05_LIBERO_ROOT/checkpoints/pi05_libero_zxw/$EXP_NAME"

find "$CKPT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n
du -sh "$CKPT_ROOT" "$HF_HOME" "$OPENPI_DATA_HOME" "$JAX_COMPILATION_CACHE_DIR"
grep -Ei 'nan|inf|out of memory|traceback|error' "$PI05_LIBERO_ROOT/logs/${EXP_NAME}.log" | tail -n 50
```

当前实现用循环下标作为 checkpoint 目录名，因此完成 30,000 次更新后，最终目录通常可能显示为 `29999`，
不要仅凭目录必须叫 `30000` 来判断失败。以日志跑满 30,000 次 update、训练进程正常退出且最终目录含
`params`、`train_state` 和 `assets` 为准：

```bash
STEP=29999  # 改成上一步实际列出的最大数字目录
test -d "$CKPT_ROOT/$STEP/params"
test -d "$CKPT_ROOT/$STEP/train_state"
test -s "$CKPT_ROOT/$STEP/assets/physical-intelligence/libero/norm_stats.json"
```

## 9. 启动策略服务

在终端 A 中使用训练环境：

```bash
cd /home/pai/zxw/openpi
source examples/libero/zxw_env.sh

export EXP_NAME="pi05_libero_full_30k_seed42"  # 与训练时一致
export STEP=29999                                # 改成实际最终 checkpoint
export CUDA_VISIBLE_DEVICES=0

CKPT="$PI05_LIBERO_ROOT/checkpoints/pi05_libero_zxw/$EXP_NAME/$STEP"
test -d "$CKPT/params"

uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_libero_zxw \
  --policy.dir "$CKPT"
```

服务默认监听 `0.0.0.0:8000`。看到模型加载完成且 server 开始监听后，再启动评测客户端。

## 10. 创建独立 LIBERO Python 3.8 评测环境

训练环境要求 Python 3.11，而当前 LIBERO 仿真依赖使用 Python 3.8，因此不要把两套依赖装进同一个 venv。

在终端 B 中执行一次：

```bash
cd /home/pai/zxw/openpi
source examples/libero/zxw_env.sh

export LIBERO_EVAL_VENV="$PI05_LIBERO_ROOT/venvs/libero_py38"
uv venv --python 3.8 "$LIBERO_EVAL_VENV"
source "$LIBERO_EVAL_VENV/bin/activate"

uv pip sync \
  examples/libero/requirements.txt \
  third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
```

每次打开新的评测终端，执行：

```bash
cd /home/pai/zxw/openpi
source examples/libero/zxw_env.sh
source "$PI05_LIBERO_ROOT/venvs/libero_py38/bin/activate"
export PYTHONPATH="$PWD/third_party/libero:$PWD/packages/openpi-client/src:${PYTHONPATH:-}"
```

`LIBERO_CONFIG_PATH` 已由 `zxw_env.sh` 指向仓库中的 `examples/libero/zxw_config/config.yaml`，因此 LIBERO 的
BDDL、初始状态和 assets 都从 `/home/pai/zxw/openpi/third_party/libero` 读取，不会在 home 之外创建配置。

先做导入检查：

```bash
python -c 'from libero.libero import benchmark, get_libero_path; print(get_libero_path("bddl_files")); print(benchmark.get_benchmark_dict().keys())'
```

## 11. 评测

### 11.1 单回合冒烟测试

先确认策略服务、图像渲染和动作链路都能工作：

```bash
MUJOCO_GL=egl python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 \
  --args.video-out-path "$PI05_LIBERO_ROOT/videos/smoke_spatial"
```

若 EGL 报错，改成：

```bash
MUJOCO_GL=glx python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 \
  --args.video-out-path "$PI05_LIBERO_ROOT/videos/smoke_spatial_glx"
```

### 11.2 四个 suite 的完整评测

正式数字必须保持：

```text
seed = 7
replan_steps = 5
num_trials_per_task = 50
resize_size = 224
```

这些正是脚本默认值。依次执行四个 suite：

```bash
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  MUJOCO_GL=egl python examples/libero/main.py \
    --args.host 127.0.0.1 \
    --args.task-suite-name "$SUITE" \
    --args.num-trials-per-task 50 \
    --args.seed 7 \
    --args.replan-steps 5 \
    --args.video-out-path "$PI05_LIBERO_ROOT/videos/$EXP_NAME/$SUITE" \
    2>&1 | tee "$PI05_LIBERO_ROOT/logs/eval_${EXP_NAME}_${SUITE}.log"
done
```

每个 suite 有 10 个任务，每任务 50 次，因此每个 suite 共 500 次，四个 suite 共 2,000 次 rollout。
日志最后的 `Total success rate` 是该 suite 的成功率。四项算术平均后与 96.85 参考值比较。

可以快速提取结果：

```bash
grep 'Total success rate:' "$PI05_LIBERO_ROOT"/logs/eval_${EXP_NAME}_*.log
```

## 12. 用官方 checkpoint 验证评测链路（推荐的控制实验）

如果自训结果明显偏低，先用官方 `pi05_libero` checkpoint 跑同一评测脚本，以区分“训练问题”和“评测环境
问题”。终端 A 改为：

```bash
cd /home/pai/zxw/openpi
source examples/libero/zxw_env.sh
export CUDA_VISIBLE_DEVICES=0

uv run scripts/serve_policy.py --env LIBERO
```

官方权重会下载到：

```text
$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero
```

评测终端 B 的命令保持不变。官方 checkpoint 若也明显低于参考值，应优先检查 MuJoCo/LIBERO 版本、相机图像、
`seed=7`、`replan_steps=5`、初始状态文件和异常日志，而不是立刻重训。

## 13. 常见问题

### 13.1 `Normalization stats not found`

确认使用的是 `pi05_libero_zxw`，并检查：

```bash
ls -l "$PI05_LIBERO_ROOT/assets/pi05_libero_zxw/physical-intelligence/libero/norm_stats.json"
```

缺失时重新执行第 6 节，且不要用官方配置名 `pi05_libero` 代替本地配置名。

### 13.2 CUDA OOM

先确认是全量微调且硬件满足大于 70 GB 的官方要求。多卡使用 FSDP；不要用降低 batch 的结果冒充严格复现。
还可确认没有其他进程占显存，并保留 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`。

### 13.3 数据或模型又写到了 `~/.cache`

通常是新终端忘记 source：

```bash
source /home/pai/zxw/openpi/examples/libero/zxw_env.sh
```

随后检查第 3 节列出的变量。不要在训练运行中直接移动正在使用的缓存。

### 13.4 Hugging Face 下载中断

保留相同的 `HF_HOME` 和 `HF_LEROBOT_HOME` 后重跑 `compute_norm_stats.py`；Hugging Face/LeRobot 会复用已经
完成的文件。先排查磁盘满、代理、证书和访问权限。

### 13.5 训练中断

不要加 `--overwrite`；使用原实验名和 `--resume`。若目录里还没有第一个有效 checkpoint，训练器会提示无法
恢复，这种情况只能从头开始一个新实验名。

### 13.6 EGL 初始化失败

先确认 GPU 驱动在评测 venv 中可见，再尝试 `MUJOCO_GL=glx`。无显示器节点若使用 GLX，还需要正确的 X11/
DISPLAY 配置。

## 14. 为什么主流程不使用 Docker

仓库官方推荐 Docker 做 LIBERO 评测，但 Docker 拉取的 CUDA 基础镜像和 build layer 会写入 Docker daemon 的
`Docker Root Dir`。普通用户仅设置 compose volume，不能保证这些大镜像位于 zxw 的个人目录。为满足“所有需要
拉取的大文件保存到 zxw 个人目录”，本文主流程采用宿主机 Python 3.11 策略服务 + zxw 目录内 Python 3.8
LIBERO venv。

如果管理员已经确认 Docker Root Dir 也位于 zxw 可用的个人存储，才建议改用
`examples/libero/README.md` 的 compose 流程。

## 15. 最终复现验收清单

- [x] 用户终端确认当前节点为 4×A800。
- [ ] 每个终端都 source 了 `examples/libero/zxw_env.sh`。
- [ ] 所有缓存变量都指向 `/home/pai/zxw/openpi_data/pi05_libero`。
- [x] `pi05_base/params` 已下载到 `OPENPI_DATA_HOME`，20 个对象、12,441,721,931 字节复核通过。
- [ ] `physical-intelligence/libero` 已下载到 `HF_LEROBOT_HOME`。
- [ ] 全量数据生成的 `norm_stats.json` 存在且 JSON 可解析。
- [ ] 使用 `pi05_libero_zxw`、batch 256、30k steps、seed 42 完成训练。
- [ ] 最终 checkpoint 同时含 `params`、`train_state`、`assets`。
- [ ] 冒烟评测能连接端口 8000，并能生成 rollout 视频。
- [ ] 四个 suite 均按 seed 7、replan 5、每任务 50 次完成。
- [ ] 保存四个成功率、平均值、训练 commit、GPU 型号和训练日志。
- [ ] 若结果异常，已用官方 `pi05_libero` checkpoint 做过同环境控制实验。

## 16. 本次对仓库及本地数据所做的全部修改

记录日期：2026-08-18（UTC）。

### 16.1 仓库工作树修改

| 文件 | 修改内容 |
| --- | --- |
| `docs/pi05_libero_finetune_zxw.md` | 新增本文：环境、下载、norm stats、单卡/4 卡全量训练、续训、服务与完整评测步骤 |
| `src/openpi/training/config.py` | 新增 `pi05_libero_zxw`；训练参数与官方 `pi05_libero` 一致，assets/checkpoint 改到 zxw 绝对路径 |
| `scripts/train.py` | JAX 编译缓存读取 `JAX_COMPILATION_CACHE_DIR`，保留 `~/.cache/jax` 作为默认值 |
| `examples/libero/zxw_env.sh` | 把 OpenPI、HF/LeRobot、uv、JAX、CUDA、Triton、W&B、临时文件等统一导向 zxw 个人目录，并配置阿里云 PyPI 镜像 |
| `examples/libero/zxw_config/config.yaml` | 把 LIBERO benchmark、BDDL、初始状态及 assets 指到 zxw 当前仓库 submodule |
| `scripts/download_pi05_base_zxw.py` | 新增当前代理兼容的并发、可续传、按对象大小校验、原子发布下载器 |
| `examples/libero/README.md` | 增加本文入口链接 |

其中 `zxw_env.sh` 后续又增加了 Hugging Face Xet 高性能并发、600 秒大文件超时和独立 Xet 缓存路径，用于
加速 LIBERO 数据集下载；同时显式设置阿里云 PyPI 为 uv 默认索引和兼容索引。针对用户终端直连
`huggingface.co` 返回 `Network is unreachable` 的情况，又固定了已验证可直连的 `https://hf-mirror.com`，并
移除了弃用的 `TRANSFORMERS_CACHE`。

`pi05_libero_zxw` 已通过 AST 对比检查：除配置名、`assets_base_dir` 和 `checkpoint_base_dir` 外，显式训练字段
与官方 `pi05_libero` 一致。

### 16.2 zxw 个人数据目录修改

已创建并写入：

```text
/home/pai/zxw/openpi_data/pi05_libero/cache/openpi/openpi-assets/checkpoints/pi05_base/params
```

该目录是完整的官方 π0.5 base Orbax 参数 checkpoint，共 20 个文件。没有启动训练、没有生成 norm stats，也没有
下载 `physical-intelligence/libero` 数据集；这些仍需按第 4、6、7 节依次完成。
