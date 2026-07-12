# T2IEval 运行说明

T2IEval 是一个统一的文生图评测框架。本仓库已经接入：

- `geneval`：使用 Mask2Former 和 CLIP 判断对象、数量、颜色、位置等能力；
- `ella`：使用 DPG-Bench 数据和本地 mPLUG VQA 评测 Prompt 遵循能力；
- `genaibench`：使用 CLIP-FlanT5 VQAScore 评测文本—图像对齐；
- `t2i_corebench`：读取 T2I-CoReBench checklist，使用本地 Qwen3.5-9B 多模态模型评分。

四个 Benchmark 共用同一个 CLI、模型接口和结果保存格式：

```bash
t2i-eval -e <benchmark> -f <config.yaml>
```

> 仓库中的正式 YAML 按当前实验服务器编写，包含 `/root/autodl-tmp/...` 绝对路径。换机器运行前，必须修改模型、数据、Judge Python 和输出目录。

## 1. 环境安装

### 1.1 基础要求

- Linux；
- Python 3.13；
- NVIDIA GPU 和可用的 CUDA 驱动；
- `uv` 包管理器；
- 足够的磁盘空间存放生成模型、评分模型和结果图片。

安装 `uv`（已经安装时跳过）：

```bash
pip install -U uv
```

### 1.2 安装主环境

```bash
cd /root/autodl-tmp/exam/T2IEval-test

uv sync --frozen
```

`--frozen` 表示严格使用仓库中的 `uv.lock`，避免依赖版本在复现过程中发生变化。

检查环境：

```bash
uv run --no-sync python -c '
import torch, transformers, diffusers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("diffusers:", diffusers.__version__)
print("cuda available:", torch.cuda.is_available())
'

uv run --no-sync t2i-eval --help
```

当前锁定环境已经验证的关键版本包括：

```text
Python       3.13
PyTorch      2.9.1 + CUDA 12.8
Transformers 4.57.3
t2v-metrics  1.1
```

### 1.3 配置缓存目录

建议将模型缓存放在项目目录之外，避免模型权重进入 Git 仓库：

```bash
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache
```

可以将上述内容写入 `~/.bashrc`，以后重新登录无需重复设置：

```bash
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache
EOF

source ~/.bashrc
```

## 2. 模型与数据准备

### 2.1 Hugging Face 登录

公开模型不需要登录。SDXL 等受限模型需要先在 Hugging Face 网页接受协议，然后执行：

```bash
uv run --no-sync hf auth login
```

登录状态检查：

```bash
uv run --no-sync hf auth whoami
```

### 2.2 生成模型

按实际评测任务下载需要的模型，不必一次下载全部模型：

```bash
# GenEval / ELLA：Stable Diffusion 1.5
uv run --no-sync hf download \
  stable-diffusion-v1-5/stable-diffusion-v1-5

# GenAI-Bench：Stable Diffusion 2.1 768
uv run --no-sync hf download \
  sd2-community/stable-diffusion-2-1

# GenAI-Bench：SDXL
uv run --no-sync hf download \
  stabilityai/stable-diffusion-xl-base-1.0
```

Qwen-Image 使用 ModelScope 下载到 YAML 中配置的本地目录：

```bash
mkdir -p /root/autodl-tmp/modelscope_cache/models/Qwen

uv run --no-sync modelscope download \
  --model Qwen/Qwen-Image \
  --local_dir /root/autodl-tmp/modelscope_cache/models/Qwen/Qwen-Image
```

### 2.3 Benchmark 评分模型

GenEval 使用 Mask2Former。正式 YAML 设置了 `detector_local_files_only: true`，因此必须提前下载：

```bash
uv run --no-sync hf download \
  facebook/mask2former-swin-small-coco-instance
```

ELLA 使用 ModelScope mPLUG VQA：

```bash
mkdir -p /root/autodl-tmp/modelscope_cache/damo

uv run --no-sync modelscope download \
  --model damo/mplug_visual-question-answering_coco_large_en \
  --local_dir /root/autodl-tmp/modelscope_cache/damo/mplug_visual-question-answering_coco_large_en
```

GenAI-Bench 使用 `clip-flant5-xxl`，并会同时使用 CLIP vision tower：

```bash
uv run --no-sync hf download \
  zhiqiulin/clip-flant5-xxl

uv run --no-sync hf download \
  openai/clip-vit-large-patch14-336
```

T2I-CoReBench 的本地 Judge：

```bash
uv run --no-sync modelscope download \
  --model Qwen/Qwen3.5-9B \
  --local_dir /root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3.5-9B
```

### 2.4 Benchmark 数据

GenEval、ELLA 和 GenAI-Bench 默认从 Hugging Face Dataset 自动读取：

| Benchmark | 数据集 |
| --- | --- |
| GenEval | `Vertsineu/geneval` |
| ELLA | `Vertsineu/ella` |
| GenAI-Bench | `Vertsineu/geneaibench` |

需要提前缓存时执行：

```bash
uv run --no-sync hf download --repo-type dataset Vertsineu/geneval
uv run --no-sync hf download --repo-type dataset Vertsineu/ella
uv run --no-sync hf download --repo-type dataset Vertsineu/geneaibench
```

T2I-CoReBench 使用官方仓库中的 12 个维度 JSON：

```bash
mkdir -p /root/autodl-tmp/exam/Benchmark_code

git clone https://github.com/KwaiVGI/T2I-CoReBench.git \
  /root/autodl-tmp/exam/Benchmark_code/T2I-CoReBench
```

确认数据目录存在：

```bash
find /root/autodl-tmp/exam/Benchmark_code/T2I-CoReBench/data \
  -maxdepth 1 -type f | sort
```

### 2.5 T2I-CoReBench Judge 隔离环境

Qwen-Image 生成环境和 Qwen3.5-9B vLLM 评分环境具有不同的 Torch/CUDA 依赖，因此 Judge 使用独立环境：

```bash
uv venv --python 3.13 /root/autodl-tmp/corebench-judge/.venv

uv pip install \
  --python /root/autodl-tmp/corebench-judge/.venv/bin/python \
  --torch-backend=auto \
  -U vllm qwen-vl-utils pillow
```

验证：

```bash
/root/autodl-tmp/corebench-judge/.venv/bin/python -c '
import torch, vllm, transformers, qwen_vl_utils
print("torch:", torch.__version__)
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
'
```

YAML 中的 `judge_python` 必须指向这个 Python：

```yaml
judge_python: /root/autodl-tmp/corebench-judge/.venv/bin/python
```

## 3. 运行评测

以下命令都从仓库根目录执行：

```bash
cd /root/autodl-tmp/exam/T2IEval-test
export HF_HOME=/root/autodl-tmp/hf_cache
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache
```

环境已经完成 `uv sync` 时，推荐使用 `uv run --no-sync`，避免运行前再次解析或下载依赖。

### 3.1 GenEval：SD1.5

```bash
uv run --no-sync t2i-eval \
  -e geneval \
  -f examples/geneval/run_geneval_sd15.yaml \
  --fail-fast
```

输出：

```text
/root/autodl-tmp/exam/results/results_geneval_sd15/
└── results_geneval_diffusers.json
```

### 3.2 ELLA / DPG-Bench：SD1.5

```bash
uv run --no-sync t2i-eval \
  -e ella \
  -f examples/ella/run_ella_sd15.yaml \
  --fail-fast
```

输出：

```text
/root/autodl-tmp/exam/results/results_ella_sd15/
└── results_ella_diffusers.json
```

### 3.3 GenAI-Bench 冒烟测试

第一次运行先生成 1 张图片，验证数据、模型和 scorer 是否能正常加载：

```bash
uv run --no-sync t2i-eval \
  -e genaibench \
  -f examples/genaibench/run_genaibench_smoke.yaml \
  --fail-fast
```

冒烟测试只验证工程链路，不能与论文指标比较。

### 3.4 GenAI-Bench：SD2.1 768

```bash
uv run --no-sync t2i-eval \
  -e genaibench \
  -f examples/genaibench/run_genaibench_sd21.yaml \
  --fail-fast
```

### 3.5 GenAI-Bench：SDXL

```bash
uv run --no-sync t2i-eval \
  -e genaibench \
  -f examples/genaibench/run_genaibench_sdxl.yaml \
  --fail-fast
```

两个正式配置分别输出到：

```text
/root/autodl-tmp/exam/results/results_genaibench_sd21/
/root/autodl-tmp/exam/results/results_genaibench_sdxl/
```

### 3.6 T2I-CoReBench 冒烟测试

冒烟配置只运行 `C-MI` 的一个 Prompt：

```bash
uv run --no-sync t2i-eval \
  -e t2i_corebench \
  -f examples/t2i_corebench/run_t2i_corebench_qwen_image_smoke.yaml \
  --fail-fast
```

确认真实图片和本地 Judge 均能运行后，再启动正式任务。

### 3.7 T2I-CoReBench：Qwen-Image 部分维度

当前正式配置评测 `C-MI`、`C-MR`、`R-LR`、`R-CR` 四个维度，每个 Prompt 生成 1 张图片：

```bash
uv run --no-sync t2i-eval \
  -e t2i_corebench \
  -f examples/t2i_corebench/run_t2i_corebench_qwen_image.yaml \
  --fail-fast
```

该配置已经启用：

```yaml
output:
  save_images: true
  resume: true
  write_artifacts: true
```

任务中断后，使用同一 YAML 再次执行相同命令即可断点续跑。框架会复用已经完整保存的图片和 Judge cache。

### 3.8 只评测已有图片

先修改：

```text
examples/t2i_corebench/run_t2i_corebench_existing_images.yaml
```

将 `image_dir` 指向图片目录。目录格式为：

```text
<image_dir>/
├── C-MI/
│   ├── C-MI-001-0.png
│   └── C-MI-001-1.png
└── R-CR/
    └── R-CR-001-0.png
```

运行：

```bash
uv run --no-sync t2i-eval \
  -e t2i_corebench \
  -f examples/t2i_corebench/run_t2i_corebench_existing_images.yaml \
  --fail-fast
```

## 4. 输出文件

每次运行至少生成一个兼容旧格式的结果文件：

```text
<output_dir>/results_<benchmark>_<model-adapter>.json
```

当 YAML 设置 `write_artifacts: true` 时，还会生成：

```text
<output_dir>/runs/<run_id>/
├── config.json          # 本次模型、Benchmark 和生成配置
├── environment.json     # Python、依赖、GPU 和 Git commit
├── metrics.json         # 结构化指标
├── samples.jsonl        # 样本级结果
├── questions.jsonl      # T2I-CoReBench 问题级结果
├── judge_cache.jsonl    # 可复用的本地 Judge 回答
├── images/              # 开启 save_images 后保存图片
└── complete.marker      # 运行成功完成标记
```

不同 Benchmark 的指标定义不同，不能直接横向比较绝对数值。复现时应同时记录 checkpoint、图片尺寸、steps、CFG、seed、图片数量和 scorer 版本。

## 5. 常见问题

| 问题 | 处理方法 |
| --- | --- |
| `hf` 或 `modelscope` 找不到 | 使用 `uv run --no-sync hf ...` 或 `uv run --no-sync modelscope ...`。 |
| 运行时重复检查或下载模型 | 确保下载和运行使用同一个 `HF_HOME`，并检查 YAML 中的本地模型路径。 |
| Hugging Face 下载较慢 | 设置 `HF_ENDPOINT=https://hf-mirror.com` 和 `HF_HUB_DISABLE_XET=1`，中断后重复原命令可续传。 |
| `403 GatedRepoError` | 在模型主页接受协议并执行 `uv run --no-sync hf auth login`；镜像仍报错时临时 `unset HF_ENDPOINT`。 |
| GenEval、ELLA 或 GenAI-Bench 缺少评分模型 | 按第 2.3 节准备 Mask2Former、mPLUG、CLIP-FlanT5 和 CLIP vision tower。 |
| 显存不足 | 先运行 smoke YAML，再酌情减小图片数量、分辨率或 scorer/Judge batch size；调试配置不能用于论文指标对比。 |
| T2I-CoReBench Judge 启动失败 | 检查 `judge_python` 是否指向独立 vLLM 环境，并从日志中查找第一个 Worker traceback。 |
| 长任务中断 | 保持相同 YAML 和输出目录重新运行；`resume: true` 时会复用已保存图片和 Judge cache。 |

## 6. 测试与进一步说明

运行项目测试：

```bash
uv run --no-sync pytest -q
```

详细参数与实现说明：

- [模型参数](docs/parameters/models.md)
- [生成参数](docs/parameters/generation.md)
- [GenEval 参数](docs/parameters/evaluators/geneval.md)
- [ELLA 参数](docs/parameters/evaluators/ella.md)
- [GenAI-Bench 参数](docs/parameters/evaluators/genaibench.md)
- [T2I-CoReBench 参数](docs/parameters/evaluators/t2i_corebench.md)
- [扩展新模型或 Benchmark](docs/guides/extending.md)
