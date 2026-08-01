import os
import torch
from transformers import TrainingArguments, Trainer

from ..config import AppConfig
from ..utils.hardware import detect_hardware
from ..utils.memory import get_memory_budget, configure_memory_guardrails
from ..models.loader import load_model_and_tokenizer
from ..models.adapter import setup_peft_adapter
from ..data.loader import load_raw_dataset

def run_classification_training(config: AppConfig):
    hw_info = detect_hardware()
    mem_profile = get_memory_budget(hw_info)
    configure_memory_guardrails(hw_info, mem_profile)

    print(f"=== Starting Sequence Classification Training on {hw_info.device_name} ===")

    model, tokenizer = load_model_and_tokenizer(config, hw_info)
    
    if config.peft.use_peft:
        model = setup_peft_adapter(model, config)

    train_raw, eval_raw = load_raw_dataset(config)

    def tokenize_fn(example):
        text = example[config.dataset.text_field] if config.dataset.text_field in example else list(example.values())[0]
        return tokenizer(text, truncation=True, max_length=config.dataset.max_seq_length, padding="max_length")

    train_ds = train_raw.map(tokenize_fn, batched=True)
    eval_ds = eval_raw.map(tokenize_fn, batched=True) if eval_raw else None

    use_fp16 = config.training.fp16 or (hw_info.supports_fp16 and not hw_info.supports_bf16)
    use_bf16 = config.training.bf16 or hw_info.supports_bf16

    args = TrainingArguments(
        output_dir=config.training.output_dir,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    print("=== Executing Classification Trainer ===")
    train_result = trainer.train()
    trainer.save_model(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)

    print(f"=== Classification Training Complete! Saved to {config.training.output_dir} ===")
    return train_result
