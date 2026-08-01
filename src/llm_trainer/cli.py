import sys
import argparse

from .config import AppConfig
from .utils.hardware import detect_hardware
from .utils.memory import get_memory_budget


def print_hardware_info():
    hw_info = detect_hardware()
    mem_profile = get_memory_budget(hw_info)

    print("\n" + "=" * 55)
    print("      UNIVERSAL LLM-TRAINER HARDWARE DIAGNOSTICS      ")
    print("=" * 55)
    print(f" Device Type            : {hw_info.device_type.upper()}")
    print(f" Device Name            : {hw_info.device_name}")
    print(f" Architecture           : {hw_info.arch}")
    print(f" NVIDIA Grace / DGX     : {hw_info.is_nvidia_grace} / {hw_info.is_dgx}")
    print(f" Apple Silicon ARM      : {hw_info.is_apple_silicon}")
    print(f" APU / Integrated GPU   : {hw_info.is_apu}")
    print(f" Device Count           : {hw_info.device_count}")
    print(f" Precision Support      : BF16={hw_info.supports_bf16}, FP16={hw_info.supports_fp16}, FP8={hw_info.supports_fp8}")
    print(f" FlashAttention Support : {hw_info.supports_flash_attn}")
    print("-" * 55)
    print(f" Total System RAM       : {mem_profile.total_ram_gb:.2f} GB")
    print(f" Available System RAM   : {mem_profile.available_ram_gb:.2f} GB")
    print(f" Total VRAM             : {mem_profile.total_vram_gb:.2f} GB")
    print(f" Available VRAM         : {mem_profile.available_vram_gb:.2f} GB")
    print(f" Memory Profile Level   : {mem_profile.profile_level.upper()}")
    print(f" Recommended Max Seq Len: {mem_profile.recommended_max_seq_len}")
    print(f" Recommended Batch Size : {mem_profile.recommended_micro_batch_size}")
    print(f" Recommended Grad Accum : {mem_profile.recommended_grad_accum_steps}")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Universal LLM Training CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── hardware-info / system-check ────────────────────────────────────
    subparsers.add_parser("hardware-info",  help="Print hardware & memory diagnostics")
    subparsers.add_parser("system-check",   help="Alias for hardware-info")

    # ── train ───────────────────────────────────────────────────────────
    train_p = subparsers.add_parser("train", help="Run training pipeline (SFT/DPO/GRPO/PPO/classification)")
    train_p.add_argument("--config", required=True, help="Path to YAML config file")

    # ── evaluate ────────────────────────────────────────────────────────
    eval_p = subparsers.add_parser(
        "evaluate",
        help="Run LM Evaluation Harness benchmarks on a trained model",
    )
    eval_p.add_argument("--config",       required=True, help="Path to YAML config file")
    eval_p.add_argument("--model-path",   default=None,  help="Override model/checkpoint path")

    # ── export (LoRA merge → HF) ────────────────────────────────────────
    export_p = subparsers.add_parser("export", help="Merge LoRA adapter into base model (HF format)")
    export_p.add_argument("--adapter-path", required=True)
    export_p.add_argument("--output-path",  required=True)

    # ── export-gguf ─────────────────────────────────────────────────────
    gguf_p = subparsers.add_parser(
        "export-gguf",
        help="Export model to GGUF format for llama.cpp / ollama / LM Studio",
    )
    gguf_p.add_argument("--model-path",  required=True, help="HF model or LoRA adapter dir")
    gguf_p.add_argument("--output-dir",  required=True, help="Output directory for .gguf files")
    gguf_p.add_argument(
        "--quant-type", default="q4_k_m",
        help="Quantisation type: f16, q8_0, q5_k_m, q4_k_m (default), q3_k_m, q2_k, …",
    )

    args = parser.parse_args()

    # ── Dispatch ────────────────────────────────────────────────────────
    if args.command in ("hardware-info", "system-check"):
        print_hardware_info()

    elif args.command == "train":
        config = AppConfig.from_yaml(args.config)
        task = config.model.task_type.lower()

        if config.grpo.enabled or task == "grpo":
            from .trainers.grpo_trainer import run_grpo_training
            run_grpo_training(config)
        elif task == "ppo":
            from .trainers.ppo_trainer import run_ppo_training
            run_ppo_training(config)
        elif task == "dpo":
            from .trainers.dpo_trainer import run_dpo_training
            run_dpo_training(config)
        elif task == "classification":
            from .trainers.classification_trainer import run_classification_training
            run_classification_training(config)
        else:
            from .trainers.sft_trainer import run_sft_training
            run_sft_training(config)

        # Post-training eval if enabled
        if config.eval.enabled:
            from .evaluators.lm_eval_harness import run_evaluation
            run_evaluation(config)

    elif args.command == "evaluate":
        config = AppConfig.from_yaml(args.config)
        from .evaluators.lm_eval_harness import run_evaluation
        run_evaluation(config, model_path=args.model_path)

    elif args.command == "export":
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"=== Merging LoRA adapter from {args.adapter_path} ===")
        import json
        with open(f"{args.adapter_path}/adapter_config.json") as f:
            base_id = json.load(f)["base_model_name_or_path"]
        base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
        merged = PeftModel.from_pretrained(base, args.adapter_path).merge_and_unload()
        merged.save_pretrained(args.output_path)
        AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True).save_pretrained(args.output_path)
        print(f"=== Merged model saved to {args.output_path} ===")

    elif args.command == "export-gguf":
        from .exporters.gguf_exporter import export_to_gguf
        export_to_gguf(
            model_path=args.model_path,
            output_dir=args.output_dir,
            quant_type=args.quant_type,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
