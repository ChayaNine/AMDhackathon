---
title: RiftTune
emoji: 🔴
colorFrom: red
colorTo: orange
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
hardware: a10g-small
license: apache-2.0
---

# RiftTune — AMD ROCm/HIP Expert

LoRA fine-tune of [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
trained on AMD Instinct MI300X. Domain-specialized for AMD ROCm and HIP GPU development.

**Side-by-side comparison:** RiftTune vs the base model on your ROCm/HIP questions.

## What it knows

- CUDA → HIP API migration (hipMalloc, hipMemcpy, hipLaunchKernelGGL, ...)
- ROCm installation, device detection, and driver issues
- Kernel optimization on AMD hardware (wavefronts, occupancy, memory bandwidth)
- Debugging HIP errors and GPU-specific issues

## Thai language

Thai responses use Qwen2.5-Coder's pre-trained multilingual capability.
Select **TH** to get explanations in Thai with code/API names kept in English.

## Model weights

- Fine-tuned adapter: [nawman0209/rifttune-7b-lora](https://huggingface.co/nawman0209/rifttune-7b-lora)
- Merged model: [nawman0209/rifttune-7b](https://huggingface.co/nawman0209/rifttune-7b)
- Training dataset: [nawman0209/rifttune-dataset](https://huggingface.co/datasets/nawman0209/rifttune-dataset)

Apache 2.0. Built for AMD Developer Hackathon — Track 2 (Fine-tuning on AMD GPUs).
