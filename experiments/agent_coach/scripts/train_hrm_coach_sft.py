#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer


def patch_hrm_prefixlm_mask_compat() -> None:
    """Bridge HRM-Text remote code across nearby Transformers builds."""
    import transformers.utils.generic as generic

    if not hasattr(generic, "split_attention_implementation"):
        def split_attention_implementation(attn_implementation):
            if isinstance(attn_implementation, dict):
                base = (
                    attn_implementation.get("")
                    or attn_implementation.get("base")
                    or attn_implementation.get("default")
                    or next(iter(attn_implementation.values()), None)
                )
                return attn_implementation, base
            return None, attn_implementation

        generic.split_attention_implementation = split_attention_implementation

    import transformers.masking_utils as masking_utils

    if getattr(masking_utils.create_causal_mask, "_hrm_coach_compat", False):
        return
    real_create_causal_mask = masking_utils.create_causal_mask

    def compat_create_causal_mask(*args, block_sequence_ids=None, or_mask_function=None, **kwargs):
        if block_sequence_ids is None:
            return real_create_causal_mask(*args, or_mask_function=or_mask_function, **kwargs)

        def prefix_block_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int):
            q_block = block_sequence_ids[batch_idx, q_idx]
            kv_block = block_sequence_ids[batch_idx, kv_idx]
            return (q_block >= 0) & (q_block == kv_block)

        if or_mask_function is not None:
            original_or_mask = or_mask_function

            def combined_or_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int):
                return prefix_block_mask(batch_idx, head_idx, q_idx, kv_idx) | original_or_mask(
                    batch_idx, head_idx, q_idx, kv_idx
                )

            prefix_or_mask = combined_or_mask
        else:
            prefix_or_mask = prefix_block_mask

        return real_create_causal_mask(*args, or_mask_function=prefix_or_mask, **kwargs)

    compat_create_causal_mask._hrm_coach_compat = True
    masking_utils.create_causal_mask = compat_create_causal_mask
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").endswith("modeling_hrm_text") and hasattr(module, "create_causal_mask"):
            module.create_causal_mask = compat_create_causal_mask


def dist_info() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank


def init_distributed() -> tuple[int, int, torch.device]:
    rank, world_size, local_rank = dist_info()
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return rank, world_size, device


class JsonlPromptDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    self.rows.append({"prompt": row["prompt"], "target": row["target"]})
        if not self.rows:
            raise ValueError(f"No examples found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.rows[idx]


class PrefixLmCollator:
    def __init__(self, tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        if self.pad_id is None:
            self.pad_id = self.eos_id

    def encode_one(self, prompt: str, target: str) -> dict[str, list[int]]:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"] + [self.eos_id]

        if len(target_ids) >= self.max_seq_len:
            target_ids = target_ids[: self.max_seq_len - 1] + [self.eos_id]
            prompt_ids = prompt_ids[-1:]
        else:
            prompt_budget = self.max_seq_len - len(target_ids)
            if len(prompt_ids) > prompt_budget:
                prompt_ids = prompt_ids[-prompt_budget:]

        input_ids = prompt_ids + target_ids
        token_type_ids = [1] * len(prompt_ids) + [0] * len(target_ids)
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }

    def __call__(self, rows: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        encoded = [self.encode_one(row["prompt"], row["target"]) for row in rows]
        max_len = max(len(row["input_ids"]) for row in encoded)
        batch = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "labels": []}

        for row in encoded:
            pad = max_len - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [self.pad_id] * pad)
            batch["attention_mask"].append(row["attention_mask"] + [0] * pad)
            batch["token_type_ids"].append(row["token_type_ids"] + [0] * pad)
            batch["labels"].append(row["labels"] + [-100] * pad)

        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def set_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def lr_at_step(step: int, max_steps: int, base_lr: float, warmup_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, max_batches: int) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        batch = move_to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = model(**batch, use_cache=False).loss
        losses.append(loss.detach())
    if was_training:
        model.train()
    if not losses:
        return float("nan")
    local = torch.stack(losses).mean()
    if dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.AVG)
    return float(local.cpu())


def save_checkpoint(model, tokenizer, output_dir: Path, name: str, rank: int) -> None:
    if rank != 0:
        return
    target = output_dir / name
    target.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    unwrapped.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Local path or Hugging Face id for an HRM-Text checkpoint.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--stop-after-seconds", type=int, default=19800)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rank, world_size, device = init_distributed()
    set_seed(args.seed, rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(json.dumps({"event": "startup", "world_size": world_size, "device": str(device), **vars(args)}, default=str))

    patch_hrm_prefixlm_mask_compat()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)

    train_ds = JsonlPromptDataset(args.data_dir / "train.jsonl")
    val_ds = JsonlPromptDataset(args.data_dir / "val.jsonl")
    collator = PrefixLmCollator(tokenizer, args.max_seq_len)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed, drop_last=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.per_device_batch_size,
        sampler=val_sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )

    start = time.time()
    global_step = 0
    accum_loss = torch.tensor(0.0, device=device)
    log_path = args.output_dir / "train_log.jsonl"

    epoch = 0
    while global_step < args.max_steps and (time.time() - start) < args.stop_after_seconds:
        train_sampler.set_epoch(epoch)
        for micro_idx, batch in enumerate(train_loader):
            if global_step >= args.max_steps or (time.time() - start) >= args.stop_after_seconds:
                break

            batch = move_to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(**batch, use_cache=False).loss
                scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()
            accum_loss += loss.detach()

            if (micro_idx + 1) % args.grad_accum_steps != 0:
                continue

            lr = lr_at_step(global_step, args.max_steps, args.lr, args.warmup_steps, args.min_lr_ratio)
            for group in optim.param_groups:
                group["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            optim.zero_grad(set_to_none=True)

            mean_loss = accum_loss / args.grad_accum_steps
            if dist.is_initialized():
                dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG)
            accum_loss.zero_()

            global_step += 1
            if rank == 0 and (global_step == 1 or global_step % args.log_steps == 0):
                row = {
                    "event": "train",
                    "step": global_step,
                    "loss": float(mean_loss.cpu()),
                    "lr": lr,
                    "elapsed_sec": round(time.time() - start, 2),
                    "examples_seen": global_step * args.grad_accum_steps * args.per_device_batch_size * world_size,
                }
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)

            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                val_loss = evaluate(model, val_loader, device, args.eval_batches)
                if rank == 0:
                    row = {"event": "eval", "step": global_step, "val_loss": val_loss, "elapsed_sec": round(time.time() - start, 2)}
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True), flush=True)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(model, tokenizer, args.output_dir, f"step-{global_step}", rank)
                if dist.is_initialized():
                    dist.barrier()

        epoch += 1

    save_checkpoint(model, tokenizer, args.output_dir, "final", rank)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print(json.dumps({"event": "finished", "step": global_step, "elapsed_sec": round(time.time() - start, 2)}), flush=True)


if __name__ == "__main__":
    main()
