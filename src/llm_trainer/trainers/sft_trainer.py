import os
from ..config import AppConfig
from ..utils.hardware import detect_hardware, HardwareInfo
from ..utils.memory import get_memory_budget, configure_memory_guardrails
from ..models.loader import load_model_and_tokenizer
from ..models.adapter import setup_peft_adapter
from ..data.loader import load_raw_dataset
from ..data.processors import (
    format_and_tokenize_dataset,
    DataCollatorForUniversalSFT,
    pack_dataset,
    PackedDataCollator,
)
from ..utils.logging_callbacks import build_callbacks, resolve_report_to

def run_sft_training(config: AppConfig):
    import torch
    from transformers import TrainingArguments, Trainer

    hw_info = detect_hardware()
    mem_profile = get_memory_budget(hw_info)
    configure_memory_guardrails(hw_info, mem_profile)

    print(f"=== Hardware Detected: {hw_info.device_name} ({hw_info.device_type.upper()}) ===")
    print(f"=== Memory Profile: {mem_profile.profile_level.upper()} (RAM: {mem_profile.total_ram_gb:.1f}GB, VRAM: {mem_profile.total_vram_gb:.1f}GB) ===")

    # Auto-adapt config parameters if requested
    if config.hardware.auto_adapt_memory:
        if mem_profile.profile_level == "low":
            config.dataset.max_seq_length = min(config.dataset.max_seq_length, mem_profile.recommended_max_seq_len)
            config.training.per_device_train_batch_size = mem_profile.recommended_micro_batch_size
            config.training.gradient_accumulation_steps = mem_profile.recommended_grad_accum_steps
            config.training.gradient_checkpointing = True
            if config.model.quantization_bit is None:
                config.model.quantization_bit = 4

    # Load Model & Tokenizer
    model, tokenizer = load_model_and_tokenizer(config, hw_info)

    # Setup PEFT adapter if enabled
    if config.peft.use_peft:
        model = setup_peft_adapter(model, config)

    # Load & preprocess datasets
    train_raw, eval_raw = load_raw_dataset(config)
    train_tok = format_and_tokenize_dataset(train_raw, tokenizer, config)
    eval_tok  = format_and_tokenize_dataset(eval_raw, tokenizer, config) if eval_raw else None

    # ------------------------------------------------------------------
    # Sequence Packing (Step 1 advancement)
    # ------------------------------------------------------------------
    use_packing = config.dataset.use_sequence_packing
    if use_packing:
        print("=== Sequence Packing ENABLED — packing dataset into full-length samples ===")
        eos_id = tokenizer.eos_token_id or 0
        train_dataset = pack_dataset(train_tok, config.dataset.max_seq_length, eos_id)
        eval_dataset  = pack_dataset(eval_tok,  config.dataset.max_seq_length, eos_id) if eval_tok else None
        data_collator = PackedDataCollator(tokenizer=tokenizer)
        n_raw = len(list(train_raw)) if hasattr(train_raw, "__len__") else "?"
        print(f"    Packed {n_raw} raw samples → {len(train_dataset)} packed samples")
    else:
        train_dataset = train_tok
        eval_dataset  = eval_tok
        data_collator = DataCollatorForUniversalSFT(tokenizer=tokenizer)

    # Mixed precision flags
    use_fp16 = config.training.fp16 or (hw_info.supports_fp16 and not hw_info.supports_bf16)
    use_bf16 = config.training.bf16 or hw_info.supports_bf16
    if hw_info.device_type == "mps":
        use_fp16 = False
        use_bf16 = False

    # Logging / experiment tracking (Step 2 advancement)
    report_to = resolve_report_to(config)
    callbacks  = build_callbacks(config)

    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_checkpointing=config.training.gradient_checkpointing,
        fp16=use_fp16,
        bf16=use_bf16,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps if eval_dataset else None,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_total_limit=2,
        remove_unused_columns=False,
        use_cpu=(hw_info.device_type in ("cpu", "arm_cpu")),
        deepspeed=config.training.deepspeed,
        fsdp=config.training.fsdp,
        report_to=report_to,
        run_name=config.logging.run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
    )

    print("=== Starting SFT Training Loop ===")
    train_result = trainer.train()
    trainer.save_model(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)

    print(f"=== SFT Training Complete! Checkpoint saved to {config.training.output_dir} ===")
    return train_result
