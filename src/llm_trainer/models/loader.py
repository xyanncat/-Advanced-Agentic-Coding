import torch
from typing import Tuple, Any, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForVision2Seq,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    BitsAndBytesConfig,
)

from ..config import AppConfig
from ..utils.hardware import HardwareInfo, detect_hardware

def load_model_and_tokenizer(config: AppConfig, hw_info: Optional[HardwareInfo] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    if hw_info is None:
        hw_info = detect_hardware()

    model_id = config.model.model_name_or_path
    task_type = config.model.task_type.lower()
    
    # Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=config.model.trust_remote_code,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    # Setup Quantization Config if requested & supported
    quantization_config = None
    if config.model.quantization_bit in (4, 8):
        if hw_info.device_type in ("cuda", "rocm"):
            if config.model.quantization_bit == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=hw_info.optimal_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Determine Model Torch Dtype
    if config.model.torch_dtype == "auto":
        torch_dtype = hw_info.optimal_dtype
    elif config.model.torch_dtype == "float16":
        torch_dtype = torch.float16
    elif config.model.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # Model Kwargs
    model_kwargs = {
        "trust_remote_code": config.model.trust_remote_code,
        "torch_dtype": torch_dtype if quantization_config is None else None,
        "quantization_config": quantization_config,
    }

    if config.model.use_flash_attention and hw_info.supports_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    elif hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        model_kwargs["attn_implementation"] = "sdpa"

    # Select Model Class
    if task_type == "seq2seq":
        model_cls = AutoModelForSeq2SeqLM
    elif task_type == "classification":
        model_cls = AutoModelForSequenceClassification
    elif task_type == "vision2seq":
        model_cls = AutoModelForVision2Seq
    else:
        model_cls = AutoModelForCausalLM

    model = model_cls.from_pretrained(model_id, **model_kwargs)
    
    return model, tokenizer
