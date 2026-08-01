"""
exporters/gguf_exporter.py

Exports a trained (or merged) HuggingFace model to GGUF format for
on-device CPU inference via llama.cpp, ollama, LM Studio, etc.

Two strategies:
  1. Auto-download llama.cpp convert script (internet required)
  2. Use a local llama.cpp checkout if LLAMA_CPP_PATH env var is set

Quantisation types supported by llama.cpp:
    f32, f16, q8_0, q6_k, q5_k_m, q5_0, q4_k_m, q4_0, q3_k_m, q2_k
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


LLAMA_CPP_CONVERT_URL = (
    "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
)

SUPPORTED_QUANT_TYPES = [
    "f32", "f16",
    "q8_0",
    "q6_k",
    "q5_k_m", "q5_0",
    "q4_k_m", "q4_0",
    "q3_k_m",
    "q2_k",
]


def _find_convert_script() -> str:
    """
    Locates the llama.cpp convert_hf_to_gguf.py script.
    Checks LLAMA_CPP_PATH env var first, then auto-downloads.
    """
    llama_path = os.environ.get("LLAMA_CPP_PATH", "")
    if llama_path:
        script = Path(llama_path) / "convert_hf_to_gguf.py"
        if script.exists():
            return str(script)

    # Download to a temp dir
    print("=== Downloading llama.cpp convert_hf_to_gguf.py ===")
    try:
        import urllib.request
        tmp_dir = Path(tempfile.gettempdir()) / "llm_trainer_gguf"
        tmp_dir.mkdir(exist_ok=True)
        script_path = tmp_dir / "convert_hf_to_gguf.py"
        if not script_path.exists():
            urllib.request.urlretrieve(LLAMA_CPP_CONVERT_URL, script_path)
        return str(script_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download llama.cpp convert script: {e}\n"
            "Set LLAMA_CPP_PATH=/path/to/llama.cpp to use a local checkout."
        )


def merge_lora_if_needed(model_path: str, output_merge_dir: str, config=None) -> str:
    """
    If model_path is a LoRA adapter, merges weights into base model first.
    Returns the path to the merged (full) model ready for GGUF conversion.
    """
    adapter_config_file = Path(model_path) / "adapter_config.json"
    if not adapter_config_file.exists():
        return model_path  # already a full model

    print(f"=== Detected LoRA adapter — merging weights before GGUF export ===")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch, json

    with open(adapter_config_file) as f:
        adapter_cfg = json.load(f)
    base_model_id = adapter_cfg.get("base_model_name_or_path", "")

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, model_path).merge_and_unload()
    os.makedirs(output_merge_dir, exist_ok=True)
    merged.save_pretrained(output_merge_dir)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.save_pretrained(output_merge_dir)
    print(f"=== Merged model saved to {output_merge_dir} ===")
    return output_merge_dir


def export_to_gguf(
    model_path: str,
    output_dir: str,
    quant_type: str = "q4_k_m",
    merge_dir: Optional[str] = None,
    outtype: str = "f16",
):
    """
    Full GGUF export pipeline:
    1. Merge LoRA if needed
    2. Convert to GGUF (FP16 base)
    3. Quantise to target quant_type
    4. Print usage instructions

    Args:
        model_path  : HuggingFace model or LoRA adapter directory.
        output_dir  : Directory where .gguf files will be written.
        quant_type  : llama.cpp quantisation type (default: q4_k_m).
        merge_dir   : Temp directory for merged model (auto if None).
        outtype     : Base output type before quantisation (f16 or f32).
    """
    if quant_type not in SUPPORTED_QUANT_TYPES:
        raise ValueError(f"quant_type must be one of: {SUPPORTED_QUANT_TYPES}")

    os.makedirs(output_dir, exist_ok=True)
    if merge_dir is None:
        merge_dir = os.path.join(output_dir, "_merged_tmp")

    # Step 1: Merge LoRA if needed
    full_model_path = merge_lora_if_needed(model_path, merge_dir)

    # Step 2: Locate convert script
    convert_script = _find_convert_script()

    # Step 3: Convert HF → GGUF (FP16 base)
    model_name = Path(full_model_path).name or "model"
    gguf_f16_path = os.path.join(output_dir, f"{model_name}-{outtype}.gguf")
    print(f"=== Converting HuggingFace model → GGUF ({outtype}) ===")
    convert_cmd = [
        sys.executable, convert_script,
        full_model_path,
        "--outfile", gguf_f16_path,
        "--outtype", outtype,
    ]
    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"GGUF conversion failed:\n{result.stdout}\n{result.stderr}"
        )
    print(f"    Base GGUF written to: {gguf_f16_path}")

    # Step 4: Quantise (skip if target == outtype)
    if quant_type not in ("f32", "f16"):
        quantise_binary = _find_quantise_binary()
        gguf_q_path = os.path.join(output_dir, f"{model_name}-{quant_type}.gguf")
        print(f"=== Quantising to {quant_type.upper()} ===")
        q_cmd = [quantise_binary, gguf_f16_path, gguf_q_path, quant_type]
        q_result = subprocess.run(q_cmd, capture_output=True, text=True)
        if q_result.returncode != 0:
            print(f"WARNING: Quantisation failed — GGUF binary may not be built.\n"
                  f"{q_result.stderr}\n"
                  f"The FP16 GGUF at {gguf_f16_path} is still usable.")
        else:
            print(f"    Quantised GGUF written to: {gguf_q_path}")
            gguf_f16_path = gguf_q_path  # point to final output
    else:
        gguf_q_path = gguf_f16_path

    print("\n=== GGUF Export Complete! ===")
    print(f"    Output file : {gguf_q_path}")
    print(f"\n  Usage with llama.cpp:")
    print(f"    ./llama-cli -m {gguf_q_path} -p \"Hello\" -n 128")
    print(f"\n  Usage with ollama:")
    print(f"    ollama create mymodel -f Modelfile  (Modelfile: FROM {gguf_q_path})")
    print(f"\n  Usage with LM Studio: drag-and-drop {gguf_q_path}")
    return gguf_q_path


def _find_quantise_binary() -> str:
    """Finds the llama.cpp quantise binary from LLAMA_CPP_PATH or PATH."""
    llama_path = os.environ.get("LLAMA_CPP_PATH", "")
    for name in ("llama-quantize", "quantize", "quantize.exe"):
        candidate = Path(llama_path) / name if llama_path else Path(name)
        if candidate.exists():
            return str(candidate)
        # Try PATH
        import shutil
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError(
        "llama.cpp quantise binary not found.\n"
        "Build it with: cd llama.cpp && cmake -B build && cmake --build build -t llama-quantize\n"
        "Then set LLAMA_CPP_PATH=/path/to/llama.cpp/build/bin"
    )
