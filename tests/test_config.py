import os
import pytest
from llm_trainer.config import AppConfig

def test_app_config_defaults():
    cfg = AppConfig()
    assert cfg.model.model_name_or_path == "SmolLM-135M"
    assert cfg.dataset.max_seq_length == 1024
    assert cfg.peft.use_peft is True
    assert cfg.training.per_device_train_batch_size == 1

def test_yaml_config_serialization(tmp_path):
    yaml_file = tmp_path / "test_cfg.yaml"
    cfg = AppConfig()
    cfg.model.model_name_or_path = "Qwen/Qwen2.5-0.5B"
    cfg.to_yaml(str(yaml_file))

    loaded_cfg = AppConfig.from_yaml(str(yaml_file))
    assert loaded_cfg.model.model_name_or_path == "Qwen/Qwen2.5-0.5B"
