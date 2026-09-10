# 微调：为什么本机跑不了，以及在租用 GPU 上怎么跑

## 一、先说结论

| 任务 | 本机（Apple M4 / 16GB / MPS） | 需要的环境 |
|---|---|---|
| Reranker 冒烟验证 | ✅ `--smoke`，batch=2 / len=256 / 3 步 | — |
| Reranker 全量微调 | ❌ | 单卡 24G+（4090 / A100） |
| Qwen3-0.6B SFT 冒烟 | ✅ `--smoke` | — |
| Qwen3-8B LoRA SFT | ❌ | 单卡 40G（A100）或 24G + param offload |
| 任何 DeepSpeed 路径 | ❌ **结构性不可用** | NVIDIA GPU |

## 二、DeepSpeed 在 macOS 上不是"慢"，是根本装不上

这一点值得说清楚，因为它经常被误认为"性能问题"：

- PyPI 上 `deepspeed` **只有 sdist，没有任何平台的预编译 wheel**，安装必然触发本地编译
- `op_builder/builder.py` 里 `darwin` / `macos` 出现 **0 次**，所有算子只对 `nvcc` / `hipcc` 构建
- DeepSpeed 自带的 MPS accelerator 自报：
  - `is_fp16_supported() = False`
  - `is_bf16_supported() = False`
  - `supported_dtypes() = [float32]`
  - `_communication_backend_name = None` ← **ZeRO 无法初始化**
- 本机 `torch.cuda.is_available() = False`，`torch.distributed.is_nccl_available() = False`

所以本仓库的做法是：**训练脚本写完整并在本机验证数据管线，全量训练在租用 GPU 上执行**。
`--smoke` 不是占位符，它真的会跑前向、反向、优化器更新，并打印 loss 序列和收到梯度的
参数张量数 —— 数据管线、标签掩码、LoRA 注入这些最容易出错的地方都被覆盖了。

## 三、Qwen3-8B 为什么放不下 16GB

```
Qwen3-8B 参数量  8,190,735,360
bf16 权重        8.19e9 × 2 bytes = 16.38 GB
本机物理内存     hw.memsize = 17,179,869,184 B = 16 GiB
```

权重本身就超过总内存，还没算优化器状态、激活值和 KV cache。
本机做推理用的是 ollama 的 `qwen3:latest`（8.2B **Q4_K_M 量化**，5.2 GB），
量化后能跑推理，但量化模型不适合做训练。

## 四、租用 GPU 完整流程

### 4.1 准备数据（在本机做，产物很小，直接传上去）

```bash
conda activate derivrag
python scripts/04_mine_qa.py --all --limit 3000        # 产出 QA 数据集
python scripts/05_mine_hard_negatives.py --n-neg 7     # 产出 reranker 训练集
python scripts/08_build_sft_data.py                    # 产出 SFT 训练集（含拒答样本）
```

产物：
- `data/qa/reranker_train.jsonl` —— `{"query","pos":[...],"neg":[...]}`
- `data/qa/sft_train.jsonl` —— `{"question","context","answer","is_refusal"}`

### 4.2 GPU 机器环境

推荐 AutoDL / Vast.ai / RunPod 的 **单卡 A100-40G**。

```bash
# CUDA 12.1 + PyTorch 2.x 的基础镜像
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.57 peft>=0.14 accelerate>=1.0 deepspeed
pip install datasets sentencepiece
```

`deepspeed` 首次 import 会编译算子，约 3-5 分钟，属正常现象。

### 4.3 Reranker 微调

```bash
deepspeed --num_gpus 1 training/reranker/train.py \
    --model BAAI/bge-reranker-v2-m3 \
    --train-file data/qa/reranker_train.jsonl \
    --output-dir output/bge-reranker-v2-m3-derivrag \
    --deepspeed training/reranker/ds_config_zero2.json \
    --epochs 2 --batch-size 4 --group-size 8 --lr 6e-6 --fp16
```

参考量级：约 1 万条训练样本、group_size=8，A100-40G 上 2 个 epoch 约 40–60 分钟。

### 4.4 Qwen3-8B LoRA SFT

```bash
deepspeed --num_gpus 1 training/qwen3_sft/train.py \
    --model Qwen/Qwen3-8B \
    --train-file data/qa/sft_train.jsonl \
    --output-dir output/qwen3-8b-derivrag-lora \
    --deepspeed training/qwen3_sft/ds_config_zero3.json \
    --epochs 3 --batch-size 1 --grad-accum 16 --lr 1e-4 --bf16
```

24G 卡（4090）上需要把 `ds_config_zero3.json` 里的
`zero_optimization.offload_param.device` 改成 `"cpu"`，速度会明显下降但能跑通。

参考量级：约 5000 条 SFT 样本，A100-40G 上 3 个 epoch 约 2–3 小时。

### 4.5 训练完拿回本机

```bash
# reranker：整个目录拷回，改 configs/config.yaml
#   models.reranker.name: output/bge-reranker-v2-m3-derivrag
# Qwen3 LoRA：合并后转 GGUF 再给 ollama
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-8B')
m = PeftModel.from_pretrained(m, 'output/qwen3-8b-derivrag-lora').merge_and_unload()
m.save_pretrained('output/qwen3-8b-derivrag-merged')
"
# 再用 llama.cpp 的 convert_hf_to_gguf.py + quantize 转 Q4_K_M，
# ollama create derivrag-qwen3 -f Modelfile
```

### 4.6 验证微调效果

微调前后必须在**同一个 Tier-1 金标集**上跑同一套评测，否则数字没有意义：

```bash
python scripts/07_ablation.py --stages plus_rerank   # 换 reranker 前后各跑一次
python scripts/06_eval.py --generation                # 换生成模型前后各跑一次
```

把两次的 `eval/reports/ablation.md` 并排放进 README —— 这才是"微调带来多少提升"
的可信证据。**不要**用训练集里的样本去评测。

## 五、成本参考

| 平台 | 卡型 | 约每小时 | 两项训练合计 |
|---|---|---|---|
| AutoDL | A100-40G | ¥6–8 | 约 3–4 小时，¥25–35 |
| Vast.ai | A100-40G | $0.8–1.5 | 约 $3–6 |
| RunPod | A100-40G | $1.2–1.9 | 约 $4–8 |

数据准备在本机完成，上传的只有几十 MB 的 jsonl，不占用计费时间。
