"""Tests for GGUF exporter helpers (Advancement 5) — no GPU/llama.cpp required."""
import pytest
from llm_trainer.exporters.gguf_exporter import SUPPORTED_QUANT_TYPES, export_to_gguf


def test_supported_quant_types_non_empty():
    assert len(SUPPORTED_QUANT_TYPES) > 0
    assert "q4_k_m" in SUPPORTED_QUANT_TYPES
    assert "f16" in SUPPORTED_QUANT_TYPES


def test_export_invalid_quant_type_raises():
    with pytest.raises(ValueError, match="quant_type must be one of"):
        export_to_gguf(
            model_path="./fake_model",
            output_dir="./fake_output",
            quant_type="invalid_quant",
        )
