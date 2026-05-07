"""RiftTune — AMD ROCm/HIP Expert (HF Spaces demo).

Side-by-side comparison: RiftTune (fine-tuned on MI300X) vs Qwen2.5-Coder-7B base.
Both models loaded in 4-bit NF4 on NVIDIA A10G (~3.8GB each, ~10GB total with overhead).

Generation is sequential on the shared GPU — PyTorch CUDA kernels serialize.
UX: RiftTune streams first (~12s), then Base streams (~12s). Total ~24s per query.
UI copy says "side-by-side" not "simultaneous" — they do NOT run in parallel on the GPU.

Hardware: a10g-small (24GB VRAM). Specified in README.md YAML.
DO NOT use bare 'a10g' — it silently falls back to CPU (2GB) and OOMs.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterator

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TUNED_MODEL_ID = "nawman0209/rifttune-7b"
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

SYSTEM_EN = (
    "You are RiftTune, an expert in AMD ROCm and HIP GPU development. "
    "Answer precisely with working HIP code and documentation citations. "
    "Never hallucinate API names — if unsure, say so."
)

SYSTEM_TH = (
    "You are RiftTune, an expert in AMD ROCm and HIP GPU development. "
    "Answer in Thai. Use Thai for all explanations. "
    "Keep code, function names, and API names in English."
)

# Base config — pad_token_id added per-call from each model's own tokenizer.
# Do NOT share tokenizers between models even if identical:
# a fine-tuned tokenizer may have vocabulary modifications that cause silent
# shape mismatches if the wrong tokenizer is used with the other model.
GENERATION_CONFIG: dict = {
    "max_new_tokens": 512,
    "temperature": 0.2,       # Low temp: deterministic technical answers
    "top_p": 0.9,
    "repetition_penalty": 1.1,  # Prevents looping on code blocks
    "do_sample": True,
}

EVAL_RESULTS_PATH = Path("data/eval_results.json")

EXAMPLES = [
    ["How do I port cudaMalloc and cudaFree to HIP?", ""],
    ["Why does hipGetDeviceCount return 0 on my AMD GPU?", ""],
    ["What is the HIP equivalent of cudaDeviceSynchronize?", ""],
    ["ROCm กับ CUDA แตกต่างกันอย่างไร? และ HIP ช่วยในการย้ายโค้ดได้อย่างไร?", ""],
    ["How do I optimize global memory bandwidth on AMD MI300X?",
     "// CUDA kernel — naive version\n__global__ void copy(float* dst, float* src, int n) {\n    int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < n) dst[i] = src[i];\n}"],
]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_models: dict = {}
_load_lock = threading.Lock()


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # fp16 preferred on A10G (not bfloat16)
    )


def _is_peft_repo(model_id: str) -> bool:
    """Check if a HF repo is a PEFT adapter (has adapter_config.json)."""
    try:
        from peft import PeftConfig
        PeftConfig.from_pretrained(model_id)
        return True
    except Exception:
        return False


def _load_one(model_id: str):
    """Load a model — handles both merged models and PEFT adapter repos."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = _bnb_config()

    if _is_peft_repo(model_id):
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def get_models():
    """Lazy-load both models on first request. Thread-safe."""
    if _models:
        return _models["tuned"], _models["tuned_tok"], _models["base"], _models["base_tok"]

    with _load_lock:
        if _models:  # double-checked inside lock
            return _models["tuned"], _models["tuned_tok"], _models["base"], _models["base_tok"]

        print(f"Loading tuned model: {TUNED_MODEL_ID}")
        tuned_model, tuned_tok = _load_one(TUNED_MODEL_ID)
        _models["tuned"] = tuned_model
        _models["tuned_tok"] = tuned_tok

        print(f"Loading base model: {BASE_MODEL_ID}")
        base_model, base_tok = _load_one(BASE_MODEL_ID)
        _models["base"] = base_model
        _models["base_tok"] = base_tok

    return _models["tuned"], _models["tuned_tok"], _models["base"], _models["base_tok"]


# ---------------------------------------------------------------------------
# Benchmark scores
# ---------------------------------------------------------------------------

def _load_scores() -> str:
    """Load eval metrics from data/eval_results.json, or return 'pending' message."""
    if not EVAL_RESULTS_PATH.exists():
        return "_Benchmark scores: pending training run._"

    try:
        data = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
        tuned = data.get("tuned", {})
        base = data.get("base", {})
        n = data.get("n_eval", "?")

        def fmt(d: dict) -> str:
            acc = d.get("api_accuracy", "?")
            rec = d.get("api_recall", "?")
            hall = round(1 - acc, 4) if isinstance(acc, float) else "?"
            return f"api_accuracy **{acc:.1%}** · hallucination_rate **{hall:.1%}** · api_recall **{rec:.1%}**"

        lines = [
            f"**Benchmark** ({n}-question eval set) &nbsp;",
            f"🔴 RiftTune &nbsp;&nbsp; {fmt(tuned)}",
            f"⬜ Base Qwen &nbsp; {fmt(base)}",
        ]
        return "  \n".join(lines)
    except Exception as e:
        return f"_Benchmark scores: error loading ({e})_"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _build_prompt(instruction: str, context: str, lang: str) -> str:
    system = SYSTEM_TH if lang == "TH" else SYSTEM_EN
    # Use a temporary tokenizer-agnostic template for prompt building.
    # The actual tokenizer.apply_chat_template() is called in generate_streaming.
    return json.dumps({"system": system, "instruction": instruction, "context": context})


def _make_prompt_str(instruction: str, context: str, lang: str, tokenizer) -> str:
    system = SYSTEM_TH if lang == "TH" else SYSTEM_EN
    user_content = instruction.strip()
    if context.strip():
        user_content = f"{user_content}\n\n```\n{context.strip()}\n```"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_streaming(model, tokenizer, prompt: str) -> Iterator[str]:
    """Stream tokens from a model. Thread-safe; daemon thread prevents zombies on cancel."""
    # Explicit field extraction — never **encoded.
    # Some tokenizer versions return unexpected fields (e.g. token_type_ids) that
    # model.generate() rejects with TypeError if passed via **kwargs.
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    thread = threading.Thread(
        target=model.generate,
        kwargs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "streamer": streamer,
            "pad_token_id": tokenizer.eos_token_id,
            **GENERATION_CONFIG,
        },
    )
    thread.daemon = True  # Prevents zombie thread if Gradio cancels mid-stream
    thread.start()

    try:
        for token in streamer:
            yield token
    finally:
        pass  # Thread exits at generation end; daemon=True handles Gradio cancels


def respond(
    instruction: str,
    context: str,
    lang: str,
) -> Iterator[tuple[str, str]]:
    """
    Generator yielding (tuned_text, base_text) pairs for Gradio streaming.

    Sequential UX:
      Phase 1 — RiftTune streams, base panel shows "Generating..."
      Phase 2 — RiftTune done, Base streams
    """
    if not instruction.strip():
        yield ("_Please enter a question._", "")
        return

    tuned_model, tuned_tok, base_model, base_tok = get_models()

    tuned_prompt = _make_prompt_str(instruction, context, lang, tuned_tok)
    base_prompt = _make_prompt_str(instruction, context, lang, base_tok)

    # --- Phase 1: stream tuned model ---
    tuned_text = ""
    for token in generate_streaming(tuned_model, tuned_tok, tuned_prompt):
        tuned_text += token
        yield (tuned_text, "_Base model generating..._")

    # --- Phase 2: stream base model ---
    base_text = ""
    for token in generate_streaming(base_model, base_tok, base_prompt):
        base_text += token
        yield (tuned_text, base_text)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

SCORES_MD = _load_scores()

with gr.Blocks(
    title="RiftTune — AMD ROCm/HIP Expert",
    theme=gr.themes.Soft(),
    css=".output-panel textarea { font-family: monospace; font-size: 13px; }",
) as demo:

    gr.Markdown(
        """# 🔴 RiftTune — AMD ROCm / HIP Expert
Fine-tuned on AMD Instinct MI300X &nbsp;·&nbsp; Open source (Apache 2.0) &nbsp;·&nbsp; Qwen2.5-Coder-7B base"""
    )

    gr.Markdown(SCORES_MD)

    with gr.Row():
        instruction = gr.Textbox(
            label="Question",
            placeholder="e.g. How do I port cudaMemcpyAsync to HIP?",
            lines=3,
            scale=5,
        )
        lang = gr.Radio(
            choices=["EN", "TH"],
            value="EN",
            label="Language",
            scale=1,
        )

    context = gr.Textbox(
        label="Code / error context (optional)",
        placeholder="Paste CUDA code, HIP error output, or stack trace here...",
        lines=4,
        visible=True,
    )

    submit_btn = gr.Button("Generate", variant="primary", size="lg")

    with gr.Row(equal_height=True):
        tuned_out = gr.Textbox(
            label="🔴 RiftTune (fine-tuned on ROCm/HIP docs)",
            lines=18,
            show_copy_button=True,
            elem_classes=["output-panel"],
        )
        base_out = gr.Textbox(
            label="⬜ Base Qwen2.5-Coder-7B-Instruct",
            lines=18,
            show_copy_button=True,
            elem_classes=["output-panel"],
        )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[instruction, context],
        label="Example questions (click to load)",
    )

    gr.Markdown(
        """---
**About:** RiftTune is a LoRA fine-tune of [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
trained on AMD Instinct MI300X using ROCm PyTorch. Training data: ROCm documentation,
HIPIFY API tables, GPUOpen blog posts, and GitHub issues from ROCm/HIP.
Dataset and adapter weights are Apache 2.0. Built for AMD Developer Hackathon — Track 2.

Thai responses use Qwen2.5-Coder's pre-trained multilingual capability — keep code and API names in English."""
    )

    submit_btn.click(
        fn=respond,
        inputs=[instruction, context, lang],
        outputs=[tuned_out, base_out],
    )
    instruction.submit(
        fn=respond,
        inputs=[instruction, context, lang],
        outputs=[tuned_out, base_out],
    )

if __name__ == "__main__":
    demo.launch()
