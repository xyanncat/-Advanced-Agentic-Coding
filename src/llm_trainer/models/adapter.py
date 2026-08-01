from typing import Optional, List
import torch
from transformers import PreTrainedModel
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)

from ..config import AppConfig

def find_all_linear_names(model: torch.nn.Module) -> List[str]:
    """Finds all linear layers in the model for LoRA target modules."""
    cls_to_find = (torch.nn.Linear,)
    try:
        import bitsandbytes as bnb
        cls_to_find = (torch.nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:
        pass

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls_to_find):
            names = name.split('.')
            lora_module_names.add(names[-1] if len(names) > 1 else names[0])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def setup_peft_adapter(model: PreTrainedModel, config: AppConfig) -> PreTrainedModel:
    if not config.peft.use_peft:
        return model

    if config.model.quantization_bit in (4, 8):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.training.gradient_checkpointing
        )

    task_type = TaskType.CAUSAL_LM
    if config.model.task_type == "seq2seq":
        task_type = TaskType.SEQ_2_SEQ_LM
    elif config.model.task_type == "classification":
        task_type = TaskType.SEQ_CLS

    target_modules = config.peft.target_modules
    if not target_modules:
        target_modules = find_all_linear_names(model)

    peft_config = LoraConfig(
        r=config.peft.lora_r,
        lora_alpha=config.peft.lora_alpha,
        lora_dropout=config.peft.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=task_type,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
