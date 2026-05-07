from unittest.mock import patch
from src.forge.forge import Forge
from src.forge.models import NvidiaGPU


def test_device_info_testing_env(monkeypatch):
    monkeypatch.setenv("APP_SETTINGS", "testing")
    f = Forge()
    f._device_info()
    assert isinstance(f.gpu_info, NvidiaGPU)


def test_device_info_production_env(monkeypatch):
    monkeypatch.setenv("APP_SETTINGS", "production")
    f = Forge()
    mock_gpu = NvidiaGPU()
    with patch.object(f.device, "get_device_info", return_value=mock_gpu):
        f._device_info()
        assert f.gpu_info == mock_gpu
    