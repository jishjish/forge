import pytest
from unittest.mock import patch, MagicMock
from forge.forge import Forge
from forge.models import NvidiaGPU


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
    

def test_run_invalid_op():
    f = Forge()
    with pytest.raises(AssertionError):
        f.run("invalid_operation")


def test_run_portfolio_op(monkeypatch):
    f = Forge()
    with patch('forge.forge.importlib.import_module') as mock_import:
        mock_import.return_value = MagicMock()
        f.run('matmul')
        mock_import.assert_called_with("forge.codegen.portfolio.op_matmul")

