#!/usr/bin/env python
"""Qwen3 条款问答 SFT（LoRA + DeepSpeed ZeRO-3）。

⚠️ 运行环境 —— 详见 training/README_GPU.md

    Qwen3-8B 的 bf16 权重是 16.38 GB，而本机 hw.memsize = 16 GiB，
    权重本身就放不下，更别说优化器状态和激活值。DeepSpeed 又完全无法在
    macOS 上安装。所以：

        本机  -> --smoke，用 Qwen3-0.6B 在 MPS 上跑 3 步验证管线
        租卡  -> 全量 LoRA SFT，A100-40G 单卡即可

训练目标是**降低幻觉**，不是教模型新知识：
    通过 system prompt 与训练样本反复强化三条行为
        1. 只依据给定条款作答
        2. 每个事实标注来源编号
        3. 无依据时明确说"根据现有条款无法确定"
    训练数据来自 QA 挖掘产出的 (问题, 条款上下文, 答案) 三元组，
    并按比例混入**拒答样本**（给无关条款，标准答案是拒答）——
    没有拒答样本，模型只会学到"永远要答"，幻觉反而更严重。

用法:
    # 本机冒烟
    python training/qwen3_sft/train.py --smoke

    # 租用 GPU
    deepspeed --num_gpus 1 training/qwen3_sft/train.py \
        --model Qwen/Qwen3-8B \
        --train-file data/qa/sft_train.jsonl \
        --output-dir output/qwen3-8b-derivrag-lora \
        --deepspeed training/qwen3_sft/ds_config_zero3.json \
        --epochs 3 --batch-size 1 --grad-accum 16 --lr 1e-4 --bf16
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("sft")

# ⚠️ peft 0.7.x 的 target-module 自动映射表里**没有任何 qwen 条目**，
#    不显式传 target_modules 会直接抛 ValueError。这里写死。
QWEN_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

SYSTEM_PROMPT = """你是衍生品交易条款检索助手。

回答规则：
1. 只能依据【参考条款】作答，不得使用先验知识补充或推测。
2. 每个事实性陈述后标注来源编号，如 [1]。
3. 数值必须逐字引用原文，不得换算或改写。
4. 条款不足以回答时，明确说明"根据现有条款无法确定"，禁止编造。"""


def _as_id_list(encoded) -> list[int]:
    """把 tokenizer 的各种返回形态统一成 list[int]。

    可能拿到: list[int] / list[list[int]] / BatchEncoding / dict / torch.Tensor
    """
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict) and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    # 批量形态取第一条
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return list(encoded)


class SFTDataset(Dataset):
    """ChatML 格式的监督微调数据集。

    只对 assistant 回复部分计算 loss（prompt 部分的标签置为 -100），
    否则模型会把大量算力浪费在学习复述条款上下文。
    """

    def __init__(self, path: str | Path, tokenizer, max_length: int = 2048) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records: list[dict] = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("question") and rec.get("answer"):
                    self.records.append(rec)

        if not self.records:
            raise ValueError(f"{path} 里没有可用样本")
        n_refusal = sum(1 for r in self.records if r.get("is_refusal"))
        logger.info(
            "载入 SFT 样本 %d 条（其中拒答样本 %d 条，占比 %.1f%%）",
            len(self.records), n_refusal, 100 * n_refusal / len(self.records),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        context = rec.get("context", "")
        user = f"【参考条款】\n{context}\n\n【问题】\n{rec['question']}\n\n请依据上述条款作答，并标注来源编号。"

        prompt_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        # transformers 5.x 的 apply_chat_template(tokenize=True) 返回的是
        # BatchEncoding 而不是 4.x 那样的 list[int]，直接拿去拼接会
        # TypeError。这里统一归一成 list[int]，两个大版本都能跑。
        prompt_ids = _as_id_list(
            self.tokenizer.apply_chat_template(
                prompt_msgs, tokenize=True, add_generation_prompt=True
            )
        )
        answer_ids = _as_id_list(
            self.tokenizer(
                rec["answer"] + self.tokenizer.eos_token, add_special_tokens=False
            )
        )

        input_ids = (prompt_ids + answer_ids)[: self.max_length]
        # prompt 部分不计 loss
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_length]
        return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_token_id: int):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3 条款问答 SFT")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--train-file", default="data/qa/sft_train.jsonl")
    ap.add_argument("--output-dir", default="output/qwen3-derivrag-lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--deepspeed", default="")
    ap.add_argument("--local_rank", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="冒烟：换成 Qwen3-0.6B，MPS 上跑 3 步验证管线（8B 在 16GB 上放不下）",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.smoke:
        args.model = "Qwen/Qwen3-0.6B"
        args.batch_size, args.grad_accum, args.max_length = 1, 1, 512
        args.epochs, args.deepspeed = 1, ""
        logger.info("=== 冒烟模式：模型换为 %s，只验证管线 ===", args.model)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(args.device)
    logger.info("设备: %s", device)

    if args.deepspeed and device != "cuda":
        logger.error(
            "指定了 --deepspeed 但当前设备是 %s。DeepSpeed 仅支持 NVIDIA GPU，"
            "macOS 上无法安装（sdist-only，op_builder 无 darwin 分支）。",
            device,
        )
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # MPS 上 bf16 训练不稳，只在 CUDA 上启用
    dtype = torch.bfloat16 if (args.bf16 and device == "cuda") else torch.float32
    # transformers 5.x 把 torch_dtype 改名为 dtype，4.x 只认 torch_dtype
    import inspect

    dtype_kw = (
        "dtype"
        if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
        else "torch_dtype"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, **{dtype_kw: dtype}
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=QWEN_LORA_TARGETS,  # 必须显式指定，见文件头注释
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = SFTDataset(args.train_file, tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    total_steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
    if args.smoke:
        total_steps = 3

    # ---------------- DeepSpeed 路径 ----------------
    if args.deepspeed:
        import deepspeed

        engine, _, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
            config=args.deepspeed,
        )
        step = 0
        for epoch in range(args.epochs):
            for batch in loader:
                batch = {k: v.to(engine.device) for k, v in batch.items()}
                loss = engine(**batch).loss
                engine.backward(loss)
                engine.step()
                step += 1
                if step % 20 == 0:
                    logger.info("epoch %d step %d loss %.4f", epoch, step, loss.item())
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("LoRA 适配器已保存 -> %s", args.output_dir)
        return 0

    # ---------------- 原生 torch 路径 ----------------
    model.to(device)
    model.train()

    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )

    step = 0
    losses: list[float] = []
    # 必须在 zero_grad 之前统计：PyTorch 2.x 的 zero_grad 默认 set_to_none=True，
    # 循环跑完再数一律是 0，会把"梯度没回传"这个真问题掩盖掉。
    lora_grad_count = 0
    done = False
    for epoch in range(args.epochs):
        if done:
            break
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()

            if (i + 1) % args.grad_accum == 0:
                if not lora_grad_count:
                    lora_grad_count = sum(
                        1
                        for n, p in model.named_parameters()
                        if "lora" in n and p.grad is not None
                    )
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                losses.append(loss.item() * args.grad_accum)
                if step % 10 == 0 or args.smoke:
                    logger.info(
                        "epoch %d step %d/%d loss %.4f", epoch, step, total_steps, losses[-1]
                    )
                if args.smoke and step >= 3:
                    done = True
                    break

    if args.smoke:
        logger.info("冒烟结果: loss 序列 %s", [round(x, 4) for x in losses])
        logger.info("收到梯度的 LoRA 张量: %d 个", lora_grad_count)
        if lora_grad_count == 0:
            logger.error(
                "冒烟失败: 没有任何 LoRA 张量收到梯度。"
                "最常见原因是 target_modules 与该模型的层名不匹配。"
            )
            return 1
        logger.info("=== 管线验证通过。Qwen3-8B 全量 SFT 请在 GPU 上执行 ===")
        return 0

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("LoRA 适配器已保存 -> %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
