import pytest
from llm_trainer.utils.memory import get_memory_budget, MemoryProfile
from llm_trainer.utils.hardware import HardwareInfo

def test_get_memory_budget():
    mem = get_memory_budget()
    assert isinstance(mem, MemoryProfile)
    assert mem.profile_level in ("low", "medium", "high")
    assert mem.total_ram_gb > 0
    assert mem.recommended_max_seq_len in (512, 2048, 4096)

def test_simulated_low_memory():
    # Simulate low hardware profile
    dummy_hw = HardwareInfo(
        device_type="cpu",
        device_name="Test CPU",
        arch="x86_64",
    )
    mem = get_memory_budget(dummy_hw)
    assert mem.profile_level in ("low", "medium", "high")
    assert isinstance(mem.to_dict(), dict)
