import pytest
from llm_trainer.data.processors import apply_prompt_template

def test_apply_prompt_template_alpaca():
    example = {"instruction": "Write a poem", "output": "Roses are red"}
    result = apply_prompt_template(example, template_style="alpaca", prompt_field="instruction", response_field="output")
    assert "### Instruction:" in result
    assert "### Response:" in result
    assert "Write a poem" in result

def test_apply_prompt_template_chatml():
    example = {"instruction": "Say hello", "output": "Hello world"}
    result = apply_prompt_template(example, template_style="chatml", prompt_field="instruction", response_field="output")
    assert "<|im_start|>user" in result
    assert "<|im_start|>assistant" in result

def test_apply_prompt_template_text():
    example = {"text": "Simple raw text line"}
    result = apply_prompt_template(example, text_field="text")
    assert result == "Simple raw text line"
