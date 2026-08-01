"""
trainers/grpo_trainer.py

Group Relative Policy Optimization (GRPO) trainer.
GRPO is a memory-efficient RLHF variant that scores multiple completions
per prompt and uses group-normalised rewards as the signal — no separate
critic/value network needed (unlike PPO).

Reference: DeepSeek-R1, Shao et al. 2024.
"""
from ..config import AppConfig
from ..utils.hardware import detect_hardware
from ..utils.memory import get_memory_budget, configure_memory_guardrails
from ..models.loader import load_model_and_tokenizer
from ..models.adapter import setup_peft_adapter
from ..data.loader import load_raw_dataset
from ..utils.logging_callbacks import build_callbacks, resolve_report_to


def _default_reward_fn(completions: list, prompts: list, **kwargs) -> list:
    """
    Rule-based reward function used when no reward model is provided.
    Awards +1.0 for non-empty completions and deducts -0.5 for very short ones.
    Replace or extend this with any domain-specific scoring logic.
    """
    rewards = []
    for comp in completions:
        text = comp.strip() if isinstance(comp, str) else ""
        if len(text) == 0:
            rewards.append(-1.0)
        elif len(text) < 10:
            rewards.append(-0.5)
        else:
            rewards.append(1.0)
    return rewards


def _load_reward_model(reward_model_path: str, hw_info):
    """Loads an external reward model (e.g. ArmoRM, PairRM) for scoring."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    print(f"=== Loading Reward Model from {reward_model_path} ===")
    rw_tokenizer = AutoTokenizer.from_pretrained(reward_model_path, trust_remote_code=True)
    rw_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        torch_dtype=hw_info.optimal_dtype if hasattr(hw_info.optimal_dtype, "itemsize") else torch.float32,
        trust_remote_code=True,
    )
    rw_model.eval()
    return rw_model, rw_tokenizer


def run_grpo_training(config: AppConfig):
    """
    Runs GRPO training using TRL's GRPOTrainer.
    Falls back to TRL's GRPOTrainer if trl>=0.12 is installed.
    """
    hw_info = detect_hardware()
    mem_profile = get_memory_budget(hw_info)
    configure_memory_guardrails(hw_info, mem_profile)

    print(f"=== GRPO Trainer | Hardware: {hw_info.device_name} ({hw_info.device_type.upper()}) ===")
    print(f"=== Memory Profile: {mem_profile.profile_level.upper()} ===")
    print(f"=== GRPO Settings: {config.grpo.num_generations} generations/prompt, KL coef={config.grpo.kl_coef} ===")

    try:
        from trl import GRPOTrainer, GRPOConfig as TrlGRPOConfig
    except ImportError:
        raise ImportError(
            "GRPO requires TRL >= 0.12. Install with: pip install 'trl>=0.12'"
        )

    # Auto-adapt to memory profile
    if config.hardware.auto_adapt_memory and mem_profile.profile_level == "low":
        config.dataset.max_seq_length = min(config.dataset.max_seq_length, mem_profile.recommended_max_seq_len)
        config.training.per_device_train_batch_size = mem_profile.recommended_micro_batch_size
        if config.model.quantization_bit is None:
            config.model.quantization_bit = 4

    model, tokenizer = load_model_and_tokenizer(config, hw_info)
    if config.peft.use_peft:
        model = setup_peft_adapter(model, config)

    train_raw, _ = load_raw_dataset(config)

    # Reward function / model
    if config.grpo.reward_model_path:
        rw_model, rw_tokenizer = _load_reward_model(config.grpo.reward_model_path, hw_info)

        def reward_fn(completions, prompts, **kwargs):
            import torch
            inputs = rw_tokenizer(
                prompts, completions,
                return_tensors="pt", padding=True, truncation=True, max_length=1024
            )
            with torch.no_grad():
                logits = rw_model(**inputs).logits.squeeze(-1)
            return logits.tolist()
    else:
        print("=== No reward model provided — using built-in rule-based reward function ===")
        reward_fn = _default_reward_fn

    # Mixed precision
    use_bf16 = config.training.bf16 or hw_info.supports_bf16
    use_fp16 = config.training.fp16 or (hw_info.supports_fp16 and not hw_info.supports_bf16)
    if hw_info.device_type == "mps":
        use_fp16 = use_bf16 = False

    report_to = resolve_report_to(config)
    callbacks = build_callbacks(config)

    grpo_config = TrlGRPOConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        max_prompt_length=config.grpo.max_prompt_length,
        max_completion_length=config.grpo.max_completion_length,
        num_generations=config.grpo.num_generations,
        kl_coef=config.grpo.kl_coef,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        report_to=report_to,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_raw,
        callbacks=callbacks if callbacks else None,
    )

    print("=== Starting GRPO Training Loop ===")
    result = trainer.train()
    trainer.save_model(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)
    print(f"=== GRPO Training Complete! Checkpoint saved to {config.training.output_dir} ===")
    return result
