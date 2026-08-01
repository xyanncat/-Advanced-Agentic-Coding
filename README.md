<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=200&section=header&text=llm-trainer&fontSize=60&fontAlignY=35&animation=twinkling&fontColor=ffffff" alt="Header Image">
  
  <p><b>Universal, Hardware-Agnostic, and Memory-Adaptive LLM Training Framework</b></p>

  <p>
    <a href="https://github.com/xyanncat/-Advanced-Agentic-Coding/releases"><img src="https://img.shields.io/github/v/release/xyanncat/-Advanced-Agentic-Coding?style=for-the-badge&color=8A2BE2" alt="Release"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+"></a>
    <a href="https://github.com/xyanncat/-Advanced-Agentic-Coding/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  </p>
</div>

> A production-ready framework for fine-tuning, preference-aligning, and training Large Language Models (LLMs), Vision-Language Models (VLMs), and Encoder/Decoder architectures across any machine.

From **8GB RAM local laptops** up to **NVIDIA Grace Hopper / DGX SuperPOD clusters**, `llm-trainer` dynamically adapts to your hardware so you never see an Out-Of-Memory (OOM) error again.

<br>

## ✨ Key Features

<table>
<tr>
<td width="50%" valign="top">
  
### 🖥️ Universal Hardware Support
No more writing custom CUDA scripts. `llm-trainer` automatically detects and optimizes for:
- 🟩 **NVIDIA GPUs & Grace CPUs**: CUDA, FlashAttention-2, FP8, DeepSpeed ZeRO-3, FSDP.
- 🍎 **Apple Silicon (M1/M2/M3)**: MPS, unified memory optimizations.
- ⚡ **ARM CPUs**: Neoverse V2, Graviton, Ampere Altra (NEON/SVE).
- 🔴 **AMD GPUs & APUs**: ROCm / HIP & SDPA attention.
- 💻 **Integrated APUs**: DirectML / CPU offloading.

</td>
<td width="50%" valign="top">

### 🚀 Advanced Training Techniques
- 📈 **SFT** with **Sequence Packing** (2-3x throughput boost).
- 🧠 **GRPO** for modern RLHF (DeepSeek-R1 style).
- 🎯 **PPO** for classical RLHF.
- ⚖️ **DPO** for alignment without reward models.
- 🛠️ **PEFT / LoRA / QLoRA / TorchAO** for efficiency on consumer hardware.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🧠 Dynamic Memory Adaptation
Auto-profiles System RAM and GPU VRAM to dynamically set batch sizes, sequence lengths, gradient accumulation steps, and CPU offloading. Seamlessly scales from **8GB RAM to 128GB+ RAM**.

</td>
<td width="50%" valign="top">

### 📊 Observability & 📦 Exporting
- 📈 Built-in **WandB** & **TensorBoard** logging.
- 🧪 **LM Eval Harness** (MMLU, HellaSwag, GSM8K).
- 💾 Export & quantize to **GGUF format** for `llama.cpp`, `ollama`, or `LM Studio`.

</td>
</tr>
</table>

---

## ⚡ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/xyanncat/-Advanced-Agentic-Coding.git
cd -Advanced-Agentic-Coding

# Install the core framework
pip install -e .

# Install optional groups:
# pip install -e .[eval]      # For LM Eval Harness
# pip install -e .[logging]   # For WandB & TensorBoard
# pip install -e .[all]       # Everything
```

### 2. Inspect System Hardware

Check what `llm-trainer` sees on your machine and how it plans to optimize memory:

```bash
python -m llm_trainer.cli hardware-info
```

### 3. Run Training

<details>
<summary><b>🔥 SFT with Sequence Packing (Max Throughput)</b></summary>

```bash
python -m llm_trainer.cli train --config configs/sft_packed.yaml
```
</details>

<details>
<summary><b>🎯 GRPO Alignment (Rule-Based Rewards)</b></summary>

```bash
python -m llm_trainer.cli train --config configs/grpo_alignment.yaml
```
</details>

<details>
<summary><b>💻 Low-Memory 8GB Local Machines (QLoRA)</b></summary>

```bash
python -m llm_trainer.cli train --config configs/sft_low_memory_8gb.yaml
```
</details>

<details>
<summary><b>🍏 Apple Silicon ARM Macs</b></summary>

```bash
python -m llm_trainer.cli train --config configs/sft_arm_mps.yaml
```
</details>

### 4. Evaluate Trained Model

Quantitatively benchmark your model against standard datasets:

```bash
python -m llm_trainer.cli evaluate --config configs/eval_benchmarks.yaml
```

### 5. Export to GGUF (for Local Inference)

Easily export your model to run in Ollama or LM Studio:

```bash
python -m llm_trainer.cli export-gguf \
  --model-path ./output \
  --output-dir ./gguf \
  --quant-type q4_k_m
```

---

## 📂 Directory Layout

<details open>
<summary><b>Click to expand project structure</b></summary>
<br>

```text
.
├── 📁 configs/                  # YAML configurations (SFT, DPO, GRPO, Eval, Export)
├── 📁 src/llm_trainer/          # Core package source code
│   ├── 📄 config.py             # Pydantic configuration schema
│   ├── 📄 cli.py                # Command-line interface
│   ├── 📁 utils/                # Hardware detection, memory logic, logging
│   ├── 📁 models/               # Model factories & PEFT adapters
│   ├── 📁 data/                 # Datasets & sequence packing processors
│   ├── 📁 trainers/             # SFT, DPO, GRPO, PPO training loops
│   ├── 📁 evaluators/           # EleutherAI eval harness integration
│   └── 📁 exporters/            # llama.cpp GGUF conversion & quantization
└── 📁 tests/                    # 20+ passing Pytest unit tests
```
</details>

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer" alt="Footer Image">
  <br>
  <i>Built for the modern AI engineering era. Open-source and hardware-agnostic.</i>
</div>
