"""
trainers/ppo_trainer.py

Proximal Policy Optimisation (PPO) trainer.
Uses TRL's PPOTrainer which requires a separate reward model.
GRPO is generally preferred for new projects (no critic network), but PPO
remains the classical RLHF approach used in InstructGPT / early ChatGPT.
"""
from ..config import AppConfig
from ..utils.hardware import detect_hardware
from ..utils.memory import get_memory_budget, configure_memory_guardrails
from ..models.loader import load_model_and_tokenizer
from ..models.adapter import setup_peft_adapter
from ..data.loader import load_raw_dataset
from ..utils.logging_callbacks import resolve_report_to


def run_ppo_training(config: AppConfig):
    hw_info = detect_hardware()
    mem_profile = get_memory_budget(hw_info)
    configure_memory_guardrails(hw_info, mem_profile)

    print(f"=== PPO Trainer | Hardware: {hw_info.device_name} ({hw_info.device_type.upper()}) ===")

    try:
        from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    except ImportError:
        raise ImportError("PPO requires TRL >= 0.8. Install with: pip install trl")

    if not config.grpo.reward_model_path:
        raise ValueError(
            "PPO requires a reward model. Set `grpo.reward_model_path` in config "
            "or switch to GRPO which supports rule-based rewards."
        )

    if config.hardware.auto_adapt_memory and mem_profile.profile_level == "low":
        config.dataset.max_seq_length = min(config.dataset.max_seq_length, mem_profile.recommended_max_seq_len)
        config.training.per_device_train_batch_size = mem_profile.recommended_micro_batch_size
        if config.model.quantization_bit is None:
            config.model.quantization_bit = 4

    # Policy model (actor) with value head
    _, tokenizer = load_model_and_tokenizer(config, hw_info)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model.model_name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Reward model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer as HFTokenizer
    import torch
    rw_tokenizer = HFTokenizer.from_pretrained(config.grpo.reward_model_path, trust_remote_code=True)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.grpo.reward_model_path, trust_remote_code=True
    )
    reward_model.eval()

    train_raw, _ = load_raw_dataset(config)

    use_bf16 = config.training.bf16 or hw_info.supports_bf16
    report_to = resolve_report_to(config)

    ppo_cfg = PPOConfig(
        output_dir=config.training.output_dir,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=int(config.training.num_train_epochs),
        bf16=use_bf16,
        report_to=report_to,
        kl_coef=config.grpo.kl_coef,
    )

    def compute_reward(queries, responses):
        inputs = rw_tokenizer(
            queries, responses, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            scores = reward_model(**inputs).logits.squeeze(-1)
        return [s.item() for s in scores]

    trainer = PPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=train_raw,
        args=ppo_cfg,
    )

    print("=== Starting PPO Training Loop ===")
    for epoch in range(int(config.training.num_train_epochs)):
        for batch in trainer.dataloader:
            queries = batch["input_ids"]
            responses = trainer.generate(queries, max_new_tokens=config.grpo.max_completion_length)
            rewards = compute_reward(
                tokenizer.batch_decode(queries, skip_special_tokens=True),
                tokenizer.batch_decode(responses, skip_special_tokens=True),
            )
            trainer.step(queries, responses, [torch.tensor(r) for r in rewards])

    trainer.save_model(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)
    print(f"=== PPO Training Complete! Saved to {config.training.output_dir} ===")
