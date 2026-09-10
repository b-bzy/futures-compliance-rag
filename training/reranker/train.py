#!/usr/bin/env python
"""BGE-Reranker 增量微调（cross-encoder）。

⚠️ 运行环境说明 —— 请先读 training/README_GPU.md

    本脚本的 DeepSpeed 路径**只能在 NVIDIA GPU 上跑**。DeepSpeed 在 macOS
    上不是"慢"，而是结构性不可用：PyPI 只有 sdist 没有任何平台的预编译轮子，
    op_builder/builder.py 里 "darwin"/"macos" 出现 0 次，只对 nvcc/hipcc 构建；
    它自带的 MPS accelerator 自报 supported_dtypes=[float32] 且
    _communication_backend_name=None，ZeRO 根本无法初始化。

    所以本机（Apple Silicon）只跑 --smoke：MPS 上 batch=2/len=256 走 3 步，
    验证数据管线、loss 能降、梯度能回传，然后把同一份代码丢到租的卡上跑全量。

训练目标:
    给定 (query, 正条款, N 个难负例)，用 InfoNCE 让模型把正例的分数
    拉到显著高于所有负例。难负例来自检索器自身的粗排结果（见
    scripts/05_mine_hard_negatives.py），这比随机负例有效得多。

用法:
    # 本机冒烟（Apple Silicon，MPS）
    python training/reranker/train.py --smoke

    # 租用 GPU 全量训练
    deepspeed --num_gpus 1 training/reranker/train.py \
        --train-file data/qa/reranker_train.jsonl \
        --output-dir output/bge-reranker-v2-m3-derivrag \
        --deepspeed training/reranker/ds_config_zero2.json \
        --epochs 2 --batch-size 4 --group-size 8 --lr 6e-6 --fp16
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("rerank-train")


# =====================================================================
@dataclass
class Sample:
    query: str
    pos: str
    negs: list[str]


class RerankDataset(Dataset):
    """每条样本 = 1 个正例 + (group_size-1) 个负例。

    负例不足时循环补齐；负例过多时随机采样，让每个 epoch 见到的
    负例组合都不同，相当于免费的数据增强。
    """

    def __init__(self, path: str | Path, group_size: int = 8, seed: int = 42) -> None:
        self.group_size = group_size
        self.rng = random.Random(seed)
        self.samples: list[Sample] = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pos_list = rec.get("pos") or []
                negs = rec.get("neg") or []
                if not rec.get("query") or not pos_list or not negs:
                    continue
                self.samples.append(Sample(rec["query"], pos_list[0], negs))

        if not self.samples:
            raise ValueError(f"{path} 里没有可用样本")
        logger.info("载入训练样本 %d 条 (group_size=%d)", len(self.samples), group_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[str, list[str]]:
        s = self.samples[idx]
        need = self.group_size - 1
        negs = list(s.negs)
        if len(negs) >= need:
            negs = self.rng.sample(negs, need)
        else:
            # 负例不够就重复采样补齐，保证 batch 内张量形状一致
            negs = [negs[i % len(negs)] for i in range(need)]
        return s.query, [s.pos, *negs]


def make_collate(tokenizer, max_length: int):
    """把 (query, [doc…]) 展开成 batch*group 个 (query, doc) 对。"""

    def collate(batch):
        queries, docs = [], []
        for q, group in batch:
            for d in group:
                queries.append(q)
                docs.append(d)
        enc = tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc

    return collate


# =====================================================================
def group_infonce(logits: torch.Tensor, group_size: int) -> torch.Tensor:
    """组内 InfoNCE。每组第 0 个是正例，目标是让它的分数最高。"""
    grouped = logits.view(-1, group_size)
    target = torch.zeros(grouped.size(0), dtype=torch.long, device=grouped.device)
    return F.cross_entropy(grouped, target)


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(description="BGE-Reranker 增量微调")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--train-file", default="data/qa/reranker_train.jsonl")
    ap.add_argument("--output-dir", default="output/bge-reranker-derivrag")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4, help="每步的 query 组数")
    ap.add_argument("--group-size", type=int, default=8, help="1 正 + (n-1) 负")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-6)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--deepspeed", default="", help="DeepSpeed 配置路径（仅 NVIDIA GPU）")
    ap.add_argument("--local_rank", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="冒烟模式：MPS/CPU 上 batch=2 len=256 跑 3 步，验证管线可用",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.smoke:
        args.batch_size, args.group_size, args.max_length = 2, 4, 256
        args.epochs, args.deepspeed = 1, ""
        logger.info("=== 冒烟模式：只验证管线，不产出可用模型 ===")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = pick_device(args.device)
    logger.info("设备: %s", device)

    if args.deepspeed and device != "cuda":
        logger.error(
            "指定了 --deepspeed 但当前设备是 %s。DeepSpeed 仅支持 NVIDIA GPU，"
            "在 macOS 上无法安装也无法运行（详见 training/README_GPU.md）。",
            device,
        )
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)

    dataset = RerankDataset(args.train_file, group_size=args.group_size, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer, args.max_length),
        drop_last=True,
    )

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.smoke:
        total_steps = 3

    # ---------------- DeepSpeed 路径 ----------------
    if args.deepspeed:
        import deepspeed

        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
            config=args.deepspeed,
        )
        device = model_engine.device
        step = 0
        for epoch in range(args.epochs):
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model_engine(**batch).logits.view(-1)
                loss = group_infonce(logits, args.group_size)
                model_engine.backward(loss)
                model_engine.step()
                step += 1
                if step % 20 == 0:
                    logger.info("epoch %d step %d loss %.4f", epoch, step, loss.item())
        model_engine.save_16bit_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("模型已保存 -> %s", args.output_dir)
        return 0

    # ---------------- 原生 torch 路径（本机 / 单卡）----------------
    model.to(device)
    model.train()

    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    # MPS 不支持 GradScaler，fp16 只在 CUDA 上启用
    use_amp = args.fp16 and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    step = 0
    losses: list[float] = []
    # 梯度张量数必须在 optimizer.zero_grad() 之前统计 —— PyTorch 2.x 的
    # zero_grad 默认 set_to_none=True，训练循环跑完再数一律是 0。
    grad_tensor_count = 0
    done = False
    for epoch in range(args.epochs):
        if done:
            break
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(**batch).logits.view(-1)
                    loss = group_infonce(logits, args.group_size) / args.grad_accum
                scaler.scale(loss).backward()
            else:
                logits = model(**batch).logits.view(-1)
                loss = group_infonce(logits, args.group_size) / args.grad_accum
                loss.backward()

            if (i + 1) % args.grad_accum == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                if not grad_tensor_count:
                    grad_tensor_count = sum(
                        1 for p in model.parameters() if p.grad is not None
                    )
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
        logger.info("收到梯度的参数张量: %d 个", grad_tensor_count)
        ok = grad_tensor_count > 0 and len(losses) >= 2 and losses[-1] < losses[0]
        if not ok:
            logger.error("冒烟失败: 梯度未回传或 loss 未下降")
            return 1
        logger.info("=== 管线验证通过。全量训练请在 NVIDIA GPU 上执行，见 README_GPU.md ===")
        return 0

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("模型已保存 -> %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
