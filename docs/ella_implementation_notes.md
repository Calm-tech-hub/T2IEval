# ELLA evaluator 实现与修改说明

本文记录将 ELLA 论文中的 DPG-Bench 实现为 T2IEval 框架内 evaluator 的修改步骤、设计理由、评测口径和验证结果。

## 1. 实现目标

目标不是接入 ELLA 文本编码器或 ELLA 生图模型，而是实现考核要求中的 DPG-Bench 评测：

1. 使用现有 `diffusers` model backend（本任务使用 SD1.5）生成图片。
2. 使用官方 mPLUG VQA 模型逐图回答 DPG 问题。
3. 应用问题依赖关系并计算总体分数。
4. 计算 Global、Entity、Attribute、Relation、Other 及二级类别分数。
5. 通过统一 `t2i-eval -e ella` 或 YAML 运行并输出 `results_ella_diffusers.json`。

整个过程不调用 ELLA 仓库的 `compute_dpg_bench.py`，外部脚本只作为评分语义参考。

## 2. 修改文件

### 新增

- `src/t2i_eval/eval/ella/__init__.py`
  - 导出 `EllaEvaluator`。
- `src/t2i_eval/eval/ella/evaluator.py`
  - 定义配置、数据转换、CSV loader、postprocessor、aggregator 和 evaluator。
- `src/t2i_eval/eval/ella/loader.py`
  - 增加 Hugging Face dataset 与本地 parquet snapshot 的离线兼容加载。
- `src/t2i_eval/eval/ella/scorer.py`
  - 封装 ModelScope mPLUG，并实现问题依赖传播。
- `tests/ella_test.py`
  - 覆盖注册、默认参数、依赖传播、分类口径和 CSV 分组。
- `examples/run_ella_sd15.yaml`
  - 提供 SD1.5 正式评测配置。
- `docs/ella_implementation_notes.md`
  - 本实现说明。

### 修改

- `src/t2i_eval/eval/__init__.py`
  - 导入 `EllaEvaluator`，确保模块加载时执行注册装饰器。
- `docs/parameters/evaluators/ella.md`
  - 将目标说明更新为实际实现参数和运行方式。

## 3. 框架执行流程

```text
CLI/YAML
  -> registry 查找 EllaEvaluator
  -> loader 读取并标准化 ELLA 数据
  -> preprocessor 按 seed 抽样（可选）
  -> DiffusersModel 生成图片
  -> model.unload() 释放 SD 显存
  -> EllaPostprocessor 加载 mPLUG 并逐问题评分
  -> aggregator 汇总总体、L1、L2 指标
  -> Runner 写入 results_ella_diffusers.json
```

### 为什么复用 SimpleEvaluator

GenEval 已经使用 `SimpleEvaluator` 表达 loader、preprocessor、generation、postprocessor 和 aggregator 生命周期。ELLA 沿用同一结构后：

- CLI、YAML、多进程切分、结果写入均由现有框架负责。
- Diffusers backend 不需要了解 DPG 问题格式。
- mPLUG 评分不会变成仓库外的第二套执行系统。
- 后续添加 evaluator 时无需修改 Runner 的 benchmark 分支。

## 4. evaluator 注册机制

`EllaEvaluator` 使用：

```python
@register_evaluator("ella")
class EllaEvaluator(SimpleEvaluator):
    ...
```

装饰器执行时将类保存为：

```python
EVALUATOR_REGISTRY["ella"] = EllaEvaluator
```

`t2i_eval.eval.__init__` 必须导入 `EllaEvaluator`，否则模块不执行，装饰器也不会注册。用户传入 `-e ella` 后，Runner 调用：

```python
registry.get_evaluator_class("ella")
```

取得类、传入 evaluator 参数并调用 `evaluate(model)`。

## 5. 数据加载与标准化

默认数据源为：

```text
Vertsineu/ella, split=validation
```

本地数据共 1065 个 prompt、14391 个问题。每条记录标准化为：

```python
(
    GenerationConfig(prompt=..., num_images_per_prompt=4, ...),
    {
        "item_id": ...,
        "prompt": ...,
        "questions": [
            {
                "qid": 1,
                "question": "...",
                "dependency": [0],
                "category_broad": "entity",
                "category_detailed": "whole",
                "tuple": "entity - whole (...)"
            }
        ]
    }
)
```

`csv_path` 可直接读取官方 `dpg_bench.csv`。CSV 每行是一个问题，因此 loader 按 `item_id` 分组；它不会复制官方脚本中误删第一条数据的 `if i == 0: continue`。

### 离线 snapshot 回退

`hf download` 会保存原始 parquet，但不会建立 `datasets` 离线模式所需的 Arrow cache。因此 loader 优先使用 `datasets.load_dataset`，连接不可用时通过 `snapshot_download(local_files_only=True)` 找到缓存中的 `validation.parquet` 并直接解析。代码不硬编码 snapshot 哈希。

## 6. 生成配置

`EllaGenerationConfig` 默认值来自项目 ELLA 参数说明：

| 参数 | 值 |
| --- | --- |
| steps | 50 |
| guidance_scale | 12.0 |
| width / height | 512 / 512 |
| num_images_per_prompt | 4 |

正式 YAML 显式写出这些参数，避免 CLI 全局 `GenerationConfig` 默认值覆盖 benchmark 默认值。

## 7. mPLUG 评分

官方评测模型为：

```text
damo/mplug_visual-question-answering_coco_large_en
```

`MPlugVQAScorer` 延迟导入 ModelScope 并将接口统一为：

```python
answer(image, question) -> str
```

答案经过 `strip().lower()`；仅 `yes` 记为 1，其余记为 0。mPLUG 在 SD 图片全部生成且 `model.unload()` 后加载，避免两个大模型同时占用 GPU。

运行 mPLUG 需要项目中的 `fairseq` fork。安装和验证命令：

```bash
uv -v pip install \
  "fairseq @ git+https://github.com/One-sixth/fairseq@44800430a728c2216fd1cf1e8daa672f50dfacba"

uv run --no-sync python -c "import fairseq; print(fairseq.__file__)"
```

## 8. 依赖传播与分数口径

每个问题保留两套分数：

- `raw_score`：mPLUG 原始 yes/no。
- `dependency_score`：父问题失败后对子问题归零的分数。

每张图的 DPG 分数是所有 `dependency_score` 的平均值；每条 prompt 的分数是其图片分数平均值；最终 `score` 是所有 prompt 分数平均值。

官方脚本的分类统计使用依赖修正前的 `raw_score`，因此 L1/L2 分类保持该口径。官方脚本在四图模式下只返回最后一张裁剪图的分类问题分数，默认配置 `category_image_policy=official_last_image` 保留该行为；可用 `all_images` 计算全部图片的分类分数。

## 9. 异常依赖处理

公开 HF 数据的第一条记录缺少 `qid=1`，但仍有 8 个问题依赖它。默认：

```text
missing_dependency_policy=zero
```

缺失父问题会让对应子问题归零而不会终止整个评测。其他选项：

- `ignore`：忽略缺失父问题。
- `error`：严格报错。

结果中的 `missing_dependency_references` 记录异常引用次数，保证问题可审计。

## 10. 输出结构

Runner 输出：

```text
results_ella_diffusers.json
```

主要指标包括：

```json
{
  "task_scores": {},
  "task_scores_percent": {},
  "category_scores_l2": {},
  "category_scores_l2_percent": {},
  "score": 0.0,
  "score_percent": 0.0,
  "num_prompts": 0,
  "num_images": 0,
  "num_question_evaluations": 0,
  "missing_dependency_references": 0,
  "category_image_policy": "official_last_image"
}
```

0-1 指标与 T2IEval/GenEval 风格一致，百分制指标便于直接和论文表格比较。

## 11. 测试与验证结果

静态检查：

```text
ruff check: passed
```

单元测试：

```text
5 passed
```

离线数据测试：

```text
1065 prompts loaded from the local HF snapshot
```

真实 mPLUG 测试：

```text
checkpoint: all keys matched
answer: yes
```

端到端冒烟测试配置：SD1.5、1 个 prompt、1 张图、2 steps、本地离线数据和本地 mPLUG。结果：

```text
error: null
score: 0.5
score_percent: 50.0
num_prompts: 1
num_images: 1
num_question_evaluations: 10
```

产物：

```text
/root/autodl-tmp/exam/results_ella_smoke/results_ella_diffusers.json
/root/autodl-tmp/exam/results_ella_smoke/samples/000000/00.png
/root/autodl-tmp/exam/results_ella_smoke/samples/000000/metadata.json
```

## 12. 正式运行

```bash
cd /root/autodl-tmp/exam/T2IEval-test

export HF_HOME=/root/autodl-tmp/hf_cache
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache

uv run --no-sync t2i-eval -f examples/run_ella_sd15.yaml
```

正式配置会评测 1065 个 prompt，每条生成 4 张图，共 4260 张图片，并对每张图片执行对应的 DPG 问题集。该运行的耗时主要来自约 57564 次 mPLUG VQA 推理。
