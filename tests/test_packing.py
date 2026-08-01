"""Tests for sequence packing (Advancement 1)."""
import pytest
from llm_trainer.data.processors import pack_dataset

def _make_samples(lengths):
    return [{"input_ids": list(range(n)), "labels": list(range(n))} for n in lengths]

def test_pack_dataset_basic():
    samples = _make_samples([100, 200, 300, 150])
    packs = pack_dataset(samples, max_seq_length=512, eos_token_id=2)
    assert len(packs) > 0
    for p in packs:
        assert len(p["input_ids"]) == 512
        assert len(p["labels"]) == 512
        assert len(p["position_ids"]) == 512

def test_pack_dataset_single_large_sample_truncated():
    samples = _make_samples([1024])
    packs = pack_dataset(samples, max_seq_length=512, eos_token_id=2)
    assert len(packs) == 1
    assert len(packs[0]["input_ids"]) == 512

def test_pack_reduces_sample_count():
    # 10 samples of 50 tokens each → should pack into ~1 pack of 512 tokens
    samples = _make_samples([50] * 10)
    packs = pack_dataset(samples, max_seq_length=512, eos_token_id=2)
    assert len(packs) < len(samples)  # packing must reduce count

def test_pack_seq_lens_sum():
    samples = _make_samples([50, 60, 70])
    packs = pack_dataset(samples, max_seq_length=512, eos_token_id=2)
    total_payload = sum(sum(p["seq_lens"]) for p in packs)
    assert total_payload == 50 + 60 + 70  # no tokens lost
