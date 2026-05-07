"""Train RiftTune: LoRA fine-tune Qwen2.5-Coder-7B-Instruct on AMD ROCm/HIP data.

ROCm-specific requirements baked in:
- model.enable_input_require_grads() before get_peft_model() — prevents zero gradients on ROCm
- bf16=True, no bitsandbytes (QLoRA excluded — bitsandbytes incompatible with ROCm 6.x)
- DataCollatorForCompletionOnlyLM — trains only on assistant tokens, not system/user

Usage:
    # Smoke test (10 steps, verifies gradients + finite loss) — run this first
    python train.py --smoke-test --steps 10

    # Full training on AMD MI300X
    python train.py \\
        --train data/train.jsonl --eval data/eval.jsonl \\
        --output ./rifttune-checkpoints \\
        --base-model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --epochs 5 --lr 1e-4 --batch-size 8 --grad-accum 2

    # Eval only — compute api_accuracy + api_recall, optionally compare against base
    python train.py --eval-only \\
        --base-model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --checkpoint ./rifttune-checkpoints/checkpoint-best \\
        --eval data/eval.jsonl --output data/eval_results.json --compare-base
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from dataset import extract_api_names, load_api_names

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are RiftTune, a domain expert in AMD ROCm and HIP GPU programming. "
    "Answer questions about HIP APIs, ROCm installation, CUDA-to-HIP porting, "
    "kernel optimization, and debugging AMD GPU workloads. "
    "When referencing API names, use exact HIP function names (e.g. hipMalloc, hipMemcpy). "
    "Be concise and accurate."
)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Single constant — must be the same for save_steps and eval_steps.
# Splitting them causes ValueError in transformers >=4.36 with load_best_model_at_end.
EVAL_INTERVAL = 50

LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Must match Qwen2.5 chat format exactly — DataCollatorForCompletionOnlyLM masks everything before this.
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            pairs.append(json.loads(line))
    return pairs


def _format_example(example: dict, tokenizer) -> dict:
    user_content = example["instruction"]
    if example.get("input", "").strip():
        user_content = f"{user_content}\n\n{example['input']}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }


def prepare_dataset(pairs: list[dict], tokenizer) -> Dataset:
    """Format pairs with Qwen chat template → Dataset with 'text' field."""
    raw = Dataset.from_list(pairs)
    return raw.map(lambda ex: _format_example(ex, tokenizer), remove_columns=raw.column_names)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model(base_model_id: str):
    print(f"Loading tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def load_peft_model(base_model_id: str, checkpoint_path: str):
    """Two-step load for eval-only: base model first, then PEFT adapter on top."""
    model, tokenizer = load_base_model(base_model_id)
    print(f"Loading PEFT adapter: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Gradient sanity check (ROCm-specific)
# ---------------------------------------------------------------------------

def _check_gradients(model, dataset: Dataset, tokenizer) -> None:
    """One forward-backward pass to confirm gradients are non-zero.

    On ROCm, forgetting model.enable_input_require_grads() before get_peft_model()
    produces all-zero gradients — training appears to run but the model never learns.
    """
    print("  Running gradient sanity check...")
    sample_text = dataset[0]["text"]
    encoded = tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )
    input_ids = encoded["input_ids"].to(model.device)
    labels = input_ids.clone()

    model.train()
    output = model(input_ids=input_ids, labels=labels)
    output.loss.backward()

    grad_norms = [
        p.grad.norm().item()
        for p in model.parameters()
        if p.grad is not None and p.requires_grad
    ]
    model.zero_grad()

    if not grad_norms or not any(g > 1e-10 for g in grad_norms):
        raise RuntimeError(
            "All gradients are zero after backward pass.\n"
            "ROCm fix: ensure model.enable_input_require_grads() is called "
            "BEFORE get_peft_model(). Check train.py line order."
        )
    print(f"  Gradient check PASSED (max_grad_norm={max(grad_norms):.2e})")


# ---------------------------------------------------------------------------
# Inference / generation
# ---------------------------------------------------------------------------

def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 512,
) -> str:
    user_content = instruction
    if input_text.strip():
        user_content = f"{instruction}\n\n{input_text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_api_metrics(
    predictions: list[str],
    references: list[str],
    api_names: set[str],
) -> dict:
    """
    api_accuracy (precision): of API names the model outputs, fraction that are real HIP names.
    api_recall: of API names in reference answers, fraction the model also produces.

    Both metrics together prevent two failure modes:
    - hallucinating wrong names (caught by api_accuracy)
    - hedging and never naming any API (caught by api_recall)
    """
    precision_num = precision_den = 0
    recall_num = recall_den = 0

    for pred, ref in zip(predictions, references):
        pred_apis = extract_api_names(pred)
        ref_apis = extract_api_names(ref)

        if pred_apis:
            precision_num += len(pred_apis & api_names)
            precision_den += len(pred_apis)

        if ref_apis:
            recall_num += len(pred_apis & ref_apis)
            recall_den += len(ref_apis)

    return {
        "api_accuracy": round(precision_num / precision_den, 4) if precision_den else 1.0,
        "api_recall": round(recall_num / recall_den, 4) if recall_den else 1.0,
    }


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def run_eval(
    model,
    tokenizer,
    eval_pairs: list[dict],
    api_names: set[str],
    output_path: Optional[Path] = None,
    compare_model=None,
    compare_tokenizer=None,
    label: str = "tuned",
    compare_label: str = "base",
) -> dict:
    print(f"\nGenerating responses for {len(eval_pairs)} eval pairs...")
    model.eval()

    predictions, references, compare_predictions = [], [], []

    for i, pair in enumerate(eval_pairs):
        if i % 10 == 0:
            print(f"  [{i}/{len(eval_pairs)}]")
        pred = generate_response(model, tokenizer, pair["instruction"], pair.get("input", ""))
        predictions.append(pred)
        references.append(pair["output"])

        if compare_model is not None:
            cpred = generate_response(
                compare_model, compare_tokenizer,
                pair["instruction"], pair.get("input", "")
            )
            compare_predictions.append(cpred)

    metrics = compute_api_metrics(predictions, references, api_names)

    results: dict = {
        label: metrics,
        "n_eval": len(eval_pairs),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if compare_model is not None:
        compare_metrics = compute_api_metrics(compare_predictions, references, api_names)
        results[compare_label] = compare_metrics
        results["improvement"] = {
            "api_accuracy_delta": round(metrics["api_accuracy"] - compare_metrics["api_accuracy"], 4),
            "api_recall_delta": round(metrics["api_recall"] - compare_metrics["api_recall"], 4),
        }

    print(f"\n=== Results: {label} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if compare_model is not None:
        print(f"\n=== Results: {compare_label} ===")
        for k, v in compare_metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"\n=== Improvement ===")
        for k, v in results["improvement"].items():
            sign = "+" if v >= 0 else ""
            print(f"  {k}: {sign}{v:.4f}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResults written to {output_path}")

    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(args) -> None:
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (no CUDA/ROCm detected)"
    print(f"Device: {device_name}")
    print(f"Base model: {args.base_model}")
    print(f"Train: {args.train}  Eval: {args.eval}")
    print(f"Output: {args.output}\n")

    model, tokenizer = load_base_model(args.base_model)

    # ROCm fix: must be called BEFORE get_peft_model() or all gradients are zero.
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    model.print_trainable_parameters()

    train_pairs = load_jsonl(Path(args.train))
    eval_pairs = load_jsonl(Path(args.eval))

    if args.smoke_test:
        train_pairs = train_pairs[:32]
        eval_pairs = eval_pairs[:8]
        print(f"\nSmoke test: {len(train_pairs)} train pairs, {len(eval_pairs)} eval pairs")

    print(f"\nPreparing datasets...")
    train_dataset = prepare_dataset(train_pairs, tokenizer)
    eval_dataset = prepare_dataset(eval_pairs, tokenizer)
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Eval:  {len(eval_dataset)} examples")

    _check_gradients(model, train_dataset, tokenizer)

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1 if args.smoke_test else args.epochs,
        max_steps=args.steps if args.smoke_test else -1,
        per_device_train_batch_size=2 if args.smoke_test else args.batch_size,
        gradient_accumulation_steps=1 if args.smoke_test else args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        save_steps=EVAL_INTERVAL,
        eval_steps=EVAL_INTERVAL,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=10,
        warmup_steps=0 if args.smoke_test else 20,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
    )

    print("\nStarting training...")
    result = trainer.train()
    loss = result.training_loss
    print(f"\nTraining complete. Final loss: {loss:.4f}")

    if args.smoke_test:
        assert 0 < loss < 100, f"Loss out of expected range: {loss}"
        assert not torch.isnan(torch.tensor(loss)), "Loss is NaN"
        print(f"Smoke test PASSED. loss={loss:.4f}")
        return

    best_dir = output_dir / "checkpoint-best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"\nCheckpoint saved to {best_dir}")
    print(f"Upload to HF Hub: huggingface-cli upload nawman0209/rifttune-7b {best_dir} .")

    api_names = load_api_names()
    if api_names:
        print("\nRunning post-training eval...")
        run_eval(
            model=model,
            tokenizer=tokenizer,
            eval_pairs=load_jsonl(Path(args.eval)),
            api_names=api_names,
            output_path=Path("data/eval_results.json"),
        )
    else:
        print("\nSkipping eval: data/hip_api_names.json not found. Run build_api_allowlist.py first.")


# ---------------------------------------------------------------------------
# Eval-only mode
# ---------------------------------------------------------------------------

def run_eval_only(args) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required with --eval-only")

    model, tokenizer = load_peft_model(args.base_model, args.checkpoint)

    compare_model = compare_tokenizer = None
    if args.compare_base:
        print(f"\nLoading base model for comparison...")
        compare_model, compare_tokenizer = load_base_model(args.base_model)

    api_names = load_api_names()
    if not api_names:
        print("WARNING: data/hip_api_names.json not found — api_accuracy/api_recall will be unreliable.")

    eval_pairs = load_jsonl(Path(args.eval))
    output_path = Path(args.output) if args.output else None

    run_eval(
        model=model,
        tokenizer=tokenizer,
        eval_pairs=eval_pairs,
        api_names=api_names,
        output_path=output_path,
        compare_model=compare_model,
        compare_tokenizer=compare_tokenizer,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train or evaluate RiftTune")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                   help="HF model ID for the base model")
    p.add_argument("--train", default="data/train.jsonl")
    p.add_argument("--eval", default="data/eval.jsonl")
    p.add_argument("--output", default="./rifttune-checkpoints",
                   help="Checkpoint output dir (training) or eval_results.json path (--eval-only)")
    p.add_argument("--checkpoint", default=None,
                   help="PEFT checkpoint directory (required for --eval-only)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--smoke-test", action="store_true",
                   help="Run 10 steps to verify training setup (fast)")
    p.add_argument("--steps", type=int, default=10,
                   help="Number of steps for --smoke-test")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training, evaluate a saved checkpoint")
    p.add_argument("--compare-base", action="store_true",
                   help="Also generate base model responses for comparison (requires --eval-only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_only:
        run_eval_only(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
