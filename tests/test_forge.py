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
    with patch("forge.forge.importlib.import_module") as mock_import:
        mock_import.return_value = MagicMock()
        f.run("matmul") 
        mock_import.assert_called_with("forge.codegen.linalg.op_matmul")

def test_run_missing_module():
    f = Forge()
    with patch("forge.forge.importlib.import_module", side_effect=ModuleNotFoundError):
        with pytest.raises(RuntimeError, match="Op module not found"):
            f.run("matmul")




# f = Forge()
# import numpy as np
# import pandas as pd
# import polars as pl
# 
# # np.ndarray - single asset
# data = np.random.uniform(100, 500, size=50).astype(np.float32)
# f.run("mean", data = data, realize=True)
# 
# # np.ndarray - multi asset
# data=np.random.uniform(100, 500, size=(5, 5)).astype(np.float32)
# f.run("log_returns", data = data, realize=True)
# f.run("mean", data = data, realize=False)
# f.run("mean", data = data, realize=False)
# f.realize()
# f.run("log_returns", data = data, realize=False)
# 
# # pd.Series - single asset
# data=pd.Series(np.random.uniform(100, 500, size=50), name="AAPL")
# f.run("mean", data = data, realize=False)
# 
# # pd.DataFrame - multi asset
# data=pd.DataFrame(np.random.uniform(100, 500, size=(50, 5)), columns=["AAPL","GOOGL","MSFT","AMZN","TSLA"])
# f.run("mean", data = data, realize=False)
# 
# # pl.Series - single asset
# data = pl.Series("AAPL", np.random.uniform(100, 500, size=50).astype(np.float32))
# f.run("mean", data = data, realize=False)
# 
# # pl.DataFrame - multi asset
# data = pl.DataFrame({ticker: np.random.uniform(100, 500, size=50).astype(np.float32) 
#             for ticker in ["AAPL","GOOGL","MSFT","AMZN","TSLA"]})
# f.run("mean", data = data, realize=False)
# 
# 
# f.run("mean", data = data, realize=False)
