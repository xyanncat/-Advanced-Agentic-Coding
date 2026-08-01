# Universal Modular LLM Training Framework (`llm-trainer`)

A production-ready, hardware-agnostic, and memory-adaptive framework for fine-tuning, preference-aligning, and training Large Language Models (LLMs), Vision-Language Models (VLMs), and Encoder/Decoder architectures across any machine — from **8GB RAM local laptops** up to **NVIDIA Grace Hopper / DGX SuperPOD clusters**.

---

## Key Features

- **Any Model Architecture**: Causal LMs (LLaMA, Qwen, Mistral, SmolLM, Phi), Seq2Seq (T5, FLAN), Encoder models (BERT, RoBERTa), and Multimodal (LLaVA).
- **Universal Hardware Support**:
  - **NVIDIA GPUs & Grace CPUs / DGX Sparks**: CUDA, FlashAttention-2, FP8 TransformerEngine, DeepSpeed ZeRO-3, FSDP.
  - **Apple Silicon ARM**: Metal Performance Shaders (`mps`), unified memory optimizations.
  - **ARM CPUs**: Neoverse V2, Graviton, Ampere Altra with NEON/SVE vectorization & bfloat16.
  - **AMD GPUs & APUs**: ROCm / HIP PyTorch engine & SDPA attention.
  - **Intel/AMD Integrated APUs**: DirectML / CPU offloading.
- **Dynamic Memory Adaptation (8GB RAM to 128GB+ RAM)**:
  - Auto-profiles System RAM and GPU VRAM to dynamically set batch sizes, sequence lengths, gradient accumulation steps, and CPU offloading to prevent OOM errors.
- **Advanced Training Techniques**:
  - **SFT (Supervised Fine-Tuning)** with **Sequence Packing** (2-3x throughput boost)
  - **GRPO (Group Relative Policy Optimization)** for modern RLHF (DeepSeek-R1 style)
  - **PPO (Proximal Policy Optimization)** for classical RLHF
  - **DPO (Direct Preference Optimization)**
  - **Text & Token Classification**
  - **PEFT / LoRA / QLoRA / TorchAO / Full Fine-Tuning**
- **Observability & Evaluation**:
  - Built-in **WandB** and **TensorBoard** logging callbacks.
  - Built-in **LM Evaluation Harness** for zero-shot/few-shot benchmarking (MMLU, HellaSwag, GSM8K).
- **Production Exporting**:
  - Full pipeline to export and quantize to **GGUF format** for local inference via `llama.cpp`, `ollama`, or `LM Studio`.

---

## Quick Start

### 1. Installation

```bash
pip install -e .
# Optional dependencies:
# pip install wandb tensorboard lm-eval>=0.4.0
```

### 2. Inspect System Hardware & Memory Profile

```bash
python -m llm_trainer.cli hardware-info
```

### 3. Run Training

#### SFT with Sequence Packing (Max Throughput):
```bash
python -m llm_trainer.cli train --config configs/sft_packed.yaml
```

#### GRPO Alignment (Rule-Based Rewards):
```bash
python -m llm_trainer.cli train --config configs/grpo_alignment.yaml
```

#### On Low-Memory 8GB Local Machines (SmolLM-1.7B / Qwen-0.5B with 4-bit QLoRA):
```bash
python -m llm_trainer.cli train --config configs/sft_low_memory_8gb.yaml
```

### 4. Evaluate Trained Model

```bash
python -m llm_trainer.cli evaluate --config configs/eval_benchmarks.yaml
```

### 5. Export to GGUF (for Local Inference)

```bash
python -m llm_trainer.cli export-gguf \
  --model-path ./output \
  --output-dir ./gguf \
  --quant-type q4_k_m
```

---

## Directory Layout

```
.
├── configs/                  # YAML configurations (SFT, DPO, GRPO, Eval, Export)
├── src/llm_trainer/          # Core package source code
│   ├── config.py             # Pydantic configuration schema
│   ├── cli.py                # Command-line interface
│   ├── utils/
│   │   ├── hardware.py       # Hardware auto-detection engine
│   │   ├── memory.py         # Memory budget & OOM protection engine
│   │   └── logging_callbacks.py # WandB & TensorBoard integration
│   ├── models/
│   │   ├── loader.py         # Model & tokenizer factory
│   │   └── adapter.py        # PEFT / LoRA / QLoRA adapters
│   ├── data/
│   │   ├── loader.py         # HF & local dataset loader
│   │   └── processors.py     # Sequence Packing & prompt templates
│   ├── trainers/
│   │   ├── sft_trainer.py    # SFT training loop
│   │   ├── dpo_trainer.py    # DPO training loop
│   │   ├── grpo_trainer.py   # GRPO training loop
│   │   └── ppo_trainer.py    # PPO training loop
│   ├── evaluators/
│   │   └── lm_eval_harness.py # EleutherAI eval harness integration
│   └── exporters/
│       └── gguf_exporter.py  # llama.cpp GGUF conversion & quantization
└── tests/                    # 20+ passing Pytest unit tests
```
