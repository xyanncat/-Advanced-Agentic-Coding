"""
evaluators/lm_eval_harness.py

Wraps EleutherAI's lm-evaluation-harness for zero-shot and few-shot
benchmarking of trained models across tasks like MMLU, HellaSwag, GSM8K,
TruthfulQA, HumanEval, etc.

Install: pip install lm-eval>=0.4.0
"""
import json
import os
from typing import Optional
from ..config import AppConfig
from ..utils.hardware import detect_hardware


def run_evaluation(config: AppConfig, model_path: Optional[str] = None):
    """
    Runs LM Evaluation Harness benchmarks against a trained checkpoint.

    Args:
        config: AppConfig with eval.tasks, eval.num_fewshot, etc.
        model_path: Override model path (defaults to config.training.output_dir)
    """
    hw_info = detect_hardware()

    try:
        import lm_eval
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        raise ImportError(
            "LM Evaluation Harness not installed.\n"
            "Install with: pip install lm-eval>=0.4.0"
        )

    checkpoint = model_path or config.training.output_dir
    tasks = config.eval.tasks
    num_fewshot = config.eval.num_fewshot
    output_path = config.eval.output_path
    batch_size = config.eval.batch_size

    print(f"=== LM Eval Harness | Model: {checkpoint} ===")
    print(f"    Tasks      : {tasks}")
    print(f"    Few-shot   : {num_fewshot}")
    print(f"    Batch size : {batch_size}")
    print(f"    Device     : {hw_info.device_type}")

    # Map hardware to lm-eval device string
    if hw_info.device_type in ("cuda", "rocm"):
        device = "cuda"
    elif hw_info.device_type == "mps":
        device = "mps"
    else:
        device = "cpu"

    # Build the HF model wrapper for lm-eval
    lm = HFLM(
        pretrained=checkpoint,
        device=device,
        batch_size=batch_size,
        trust_remote_code=config.model.trust_remote_code,
    )

    # Run evaluation
    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        log_samples=False,
    )

    # Save results to JSON
    os.makedirs(output_path, exist_ok=True)
    results_file = os.path.join(output_path, "eval_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results["results"], f, indent=2)

    print(f"\n=== Evaluation Results ===")
    for task_name, task_results in results["results"].items():
        print(f"  {task_name}:")
        for metric, value in task_results.items():
            if isinstance(value, float):
                print(f"    {metric:30s}: {value:.4f}")
    print(f"\n=== Results saved to {results_file} ===")
    return results
