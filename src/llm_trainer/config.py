import os
import yaml
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    model_name_or_path: str = Field(default="SmolLM-135M", description="HuggingFace model ID or local directory")
    task_type: str = Field(default="causal_lm", description="causal_lm, seq2seq, classification, vision2seq")
    torch_dtype: str = Field(default="auto", description="auto, float16, bfloat16, float32")
    quantization_bit: Optional[int] = Field(default=None, description="None, 4, 8")
    use_flash_attention: bool = Field(default=False, description="Enable FlashAttention-2 if hardware supports")
    trust_remote_code: bool = Field(default=True, description="Trust remote code for custom architectures")

class DatasetConfig(BaseModel):
    dataset_name_or_path: str = Field(default="tatsu-lab/alpaca", description="HuggingFace dataset or file path")
    dataset_config_name: Optional[str] = Field(default=None, description="Sub-dataset configuration name")
    text_field: str = Field(default="text", description="Primary text field for training")
    prompt_field: Optional[str] = Field(default=None, description="Prompt field for instruction/DPO data")
    response_field: Optional[str] = Field(default=None, description="Response field for instruction data")
    chosen_field: Optional[str] = Field(default="chosen", description="Chosen response field for DPO")
    rejected_field: Optional[str] = Field(default="rejected", description="Rejected response field for DPO")
    train_split: str = Field(default="train", description="Dataset split for training")
    eval_split: Optional[str] = Field(default="validation", description="Dataset split for evaluation")
    streaming: bool = Field(default=False, description="Stream dataset to save local disk/RAM")
    max_seq_length: int = Field(default=1024, description="Maximum sequence token length")
    chat_template: Optional[str] = Field(default=None, description="Chat template style (alpaca, chatml, llama3)")
    use_sequence_packing: bool = Field(default=False, description="Pack multiple sequences per sample to eliminate padding waste")

class PEFTConfig(BaseModel):
    use_peft: bool = Field(default=True, description="Enable Parameter-Efficient Fine-Tuning")
    peft_type: str = Field(default="LORA", description="LORA, QLORA, PREFIX_TUNING, PROMPT_TUNING")
    lora_r: int = Field(default=8, description="LoRA rank dimension")
    lora_alpha: int = Field(default=16, description="LoRA scaling alpha")
    lora_dropout: float = Field(default=0.05, description="LoRA dropout rate")
    target_modules: Optional[List[str]] = Field(default=None, description="Target modules for LoRA injection")

class TrainingConfig(BaseModel):
    output_dir: str = Field(default="./output", description="Directory to save model checkpoints")
    learning_rate: float = Field(default=2e-4, description="Training initial learning rate")
    num_train_epochs: float = Field(default=1.0, description="Total training epochs")
    max_steps: int = Field(default=-1, description="Override total training steps (-1 uses epochs)")
    per_device_train_batch_size: int = Field(default=1, description="Micro train batch size per GPU/CPU")
    per_device_eval_batch_size: int = Field(default=1, description="Micro eval batch size per GPU/CPU")
    gradient_accumulation_steps: int = Field(default=8, description="Gradient accumulation steps")
    gradient_checkpointing: bool = Field(default=True, description="Enable gradient checkpointing to save VRAM")
    fp16: bool = Field(default=False, description="Use FP16 mixed precision")
    bf16: bool = Field(default=False, description="Use BF16 mixed precision")
    logging_steps: int = Field(default=10, description="Steps interval for logging loss")
    save_steps: int = Field(default=50, description="Steps interval for saving checkpoint")
    eval_steps: int = Field(default=50, description="Steps interval for evaluation")
    deepspeed: Optional[str] = Field(default=None, description="Path to DeepSpeed JSON configuration")
    fsdp: Optional[str] = Field(default=None, description="FSDP strategy flags")
    warmup_ratio: float = Field(default=0.03, description="Fraction of steps for LR warmup")
    lr_scheduler_type: str = Field(default="cosine", description="LR scheduler: cosine, linear, constant")
    report_to: Optional[str] = Field(default=None, description="Logging backends: wandb, tensorboard, none")

class LoggingConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable experiment tracking")
    backend: str = Field(default="tensorboard", description="wandb or tensorboard")
    project_name: str = Field(default="llm-trainer", description="WandB project name")
    run_name: Optional[str] = Field(default=None, description="Experiment run name")
    tags: List[str] = Field(default_factory=list, description="WandB tags")
    log_model: bool = Field(default=False, description="Upload model artifacts to WandB")
    tensorboard_dir: str = Field(default="./runs", description="TensorBoard log directory")

class EvalConfig(BaseModel):
    enabled: bool = Field(default=False, description="Run LM Eval Harness benchmarks after training")
    tasks: List[str] = Field(default_factory=lambda: ["hellaswag", "mmlu", "gsm8k"], description="lm-eval task names")
    num_fewshot: int = Field(default=0, description="Number of few-shot examples")
    output_path: str = Field(default="./eval_results", description="Where to write benchmark JSON results")
    batch_size: int = Field(default=4, description="Batch size for evaluation")

class GRPOConfig(BaseModel):
    enabled: bool = Field(default=False, description="Use GRPO reinforcement training instead of SFT")
    num_generations: int = Field(default=4, description="Rollouts per prompt for GRPO")
    reward_model_path: Optional[str] = Field(default=None, description="Path/ID of reward model; None = rule-based")
    kl_coef: float = Field(default=0.05, description="KL penalty coefficient")
    max_prompt_length: int = Field(default=512, description="Max token length of prompt")
    max_completion_length: int = Field(default=512, description="Max token length of completion")

class HardwareMemoryConfig(BaseModel):
    auto_adapt_memory: bool = Field(default=True, description="Automatically adapt batch size & seq len to System RAM/VRAM")
    device_target: str = Field(default="auto", description="auto, cuda, rocm, mps, arm_cpu, cpu")
    max_ram_usage_gb: Optional[float] = Field(default=None, description="Hard cap on System RAM usage")

class AppConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    peft: PEFTConfig = Field(default_factory=PEFTConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    hardware: HardwareMemoryConfig = Field(default_factory=HardwareMemoryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
