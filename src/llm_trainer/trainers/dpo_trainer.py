import os
import torch
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig

from ..config import AppConfig
from ..utils.hardware import detect_hardware
from ..utils.memory import get_memory_budget, configure_memory_guardrails
from ..models.loader import load_model_and_tokenizer
from ..models.adapter import setup_peft_adapter
from ..data.loader import load_raw_dataset

def run_dpo_training(config: AppConfig):
    hw_info = detect_hardware()
    mem_profile = get_memory_budget(hw_info)
    configure_memory_guardrails(hw_info, mem_profile)

    print(f"=== Starting DPO Preference Training on {hw_info.device_name} ===")

    # Load Model & Tokenizer
    model, tokenizer = load_model_and_tokenizer(config, hw_info)
    
    if config.peft.use_peft:
        model = setup_peft_adapter(model, config)

    train_raw, eval_raw = load_raw_dataset(config)

    use_fp16 = config.training.fp16 or (hw_info.supports_fp16 and not hw_info.supports_bf16)
    use_bf16 = config.training.bf16 or hw_info.supports_bf16

    dpo_args = DPOConfig(
        output_dir=config.training.output_dir,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_length=config.dataset.max_seq_length,
        max_prompt_length=config.dataset.max_seq_length // 2,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_raw,
        eval_dataset=eval_raw,
        tokenizer=tokenizer,
    )

    print("=== Executing DPO Trainer ===")
    train_result = trainer.train()
    trainer.save_model(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)

    print(f"=== DPO Training Complete! Output saved to {config.training.output_dir} ===")
    return train_result
