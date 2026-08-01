import pytest
from llm_trainer.utils.hardware import detect_hardware, HardwareInfo

def test_detect_hardware():
    hw = detect_hardware()
    assert isinstance(hw, HardwareInfo)
    assert hw.device_type in ("cuda", "rocm", "mps", "arm_cpu", "cpu", "xpu")
    assert isinstance(hw.device_name, str)
    assert isinstance(hw.to_dict(), dict)
