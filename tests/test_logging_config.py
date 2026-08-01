"""Tests for LoggingConfig and resolve_report_to (Advancement 2)."""
import pytest
from llm_trainer.config import AppConfig
from llm_trainer.utils.logging_callbacks import resolve_report_to


def test_logging_disabled_by_default():
    cfg = AppConfig()
    assert cfg.logging.enabled is False
    assert resolve_report_to(cfg) == "none"


def test_logging_backend_wandb():
    cfg = AppConfig()
    cfg.logging.enabled = True
    cfg.logging.backend = "wandb"
    assert resolve_report_to(cfg) == "wandb"


def test_logging_backend_tensorboard():
    cfg = AppConfig()
    cfg.logging.enabled = True
    cfg.logging.backend = "tensorboard"
    assert resolve_report_to(cfg) == "tensorboard"


def test_training_report_to_overrides_logging():
    cfg = AppConfig()
    cfg.logging.enabled = True
    cfg.logging.backend = "wandb"
    cfg.training.report_to = "tensorboard"   # explicit override wins
    assert resolve_report_to(cfg) == "tensorboard"


def test_grpo_config_defaults():
    cfg = AppConfig()
    assert cfg.grpo.num_generations == 4
    assert cfg.grpo.kl_coef == 0.05
    assert cfg.grpo.reward_model_path is None


def test_eval_config_defaults():
    cfg = AppConfig()
    assert cfg.eval.enabled is False
    assert "hellaswag" in cfg.eval.tasks
    assert cfg.eval.num_fewshot == 0
