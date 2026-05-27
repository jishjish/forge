import os
import importlib
from rich import print
from pathlib import Path
from .helpers import DEBUG
import forge.models as _models
from dotenv import load_dotenv
from .device.device import Device
from forge.codegen.compile import Compile
from utils import available_gpu_models, available_portfolio_ops, available_linalg_ops, get_op_import_path
load_dotenv()

class Forge:
    _gpu_models = available_gpu_models()
    _portfolio_ops = available_portfolio_ops()
    _linalg_ops = available_linalg_ops()

    def __init__(self):
        self.device = Device()
        self.gpu_info = None
        self._device_info()

    @classmethod
    def supported_gpus(cls): print([m.__name__ for m in cls._gpu_models])

    @classmethod
    def supported_ops(cls): print({"portfolio_ops": cls._portfolio_ops, "linalg_ops": cls._linalg_ops})

    def _device_info(self):
        """ returns corresponding pydantic GPU model; (Forge --> Device --> get_device_info() --> gpu ops)"""
        if os.getenv("APP_SETTINGS", "testing") == "production":
            gpu_info = self.device.get_device_info()
            assert isinstance(gpu_info, tuple(self._gpu_models)), f"Unsupported GPU: {type(gpu_info).__name__}. Supported: {[c.__name__ for c in self._gpu_models]}"
            self.gpu_info = gpu_info
            if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] | [dim]Device initialized:[/dim] {self.gpu_info}")
        else: 
            self.gpu_info = _models.NvidiaGPU()
            if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] | [white]env set to testing, returning base NVIDIA model[/white]")
    
    def run(self, op: str, realize: bool = True, **kwargs):
        assert op in self._portfolio_ops or op in self._linalg_ops, f"{op} not found. Supported ops: {self._portfolio_ops + self._linalg_ops}"
        import_path = get_op_import_path(op) # import based on provided op (thru portfolio or linalg)

        try: decomp_pipeline = getattr(importlib.import_module(import_path), "PIPELINE")
        except ModuleNotFoundError: raise RuntimeError(f"Error retrieving decomposition pipeline: {import_path}")
        if DEBUG >= 1:
            print(f"[dim]forge ({Path(__file__).name})[/dim] | Data realization: {realize}")
            print(f"[dim]forge ({Path(__file__).name})[/dim] | Import path: {import_path} for {op} operation")
            print(f"[dim]forge ({Path(__file__).name})[/dim] | Kwargs: {kwargs}")
        # start handling of compilations
        assert self.gpu_info is not None, "Device not initialized"
        comp = Compile(self.gpu_info, import_path, op, decomp_pipeline)
        comp.generate(realize=realize, **kwargs) 

if __name__ == "__main__":
    f = Forge()
    import numpy as np
    import pandas as pd
    import polars as pl

    # np.ndarray - single asset
    # data = np.random.uniform(100, 500, size=50).astype(np.float32)
    # f.run("mean", data = data, realize=True)

    # np.ndarray - multi asset
    data=np.random.uniform(100, 500, size=(5, 5)).astype(np.float32)
    # f.run("log_returns", data = data, realize=True)
    f.run("mean", data = data, realize=True)
# 
#     # pd.Series - single asset
#     data=pd.Series(np.random.uniform(100, 500, size=50), name="AAPL")
#     f.run("mean", data = data, realize=False)
# 
#     # pd.DataFrame - multi asset
#     data=pd.DataFrame(np.random.uniform(100, 500, size=(50, 5)), columns=["AAPL","GOOGL","MSFT","AMZN","TSLA"])
#     f.run("mean", data = data, realize=False)
# 
#     # pl.Series - single asset
#     data = pl.Series("AAPL", np.random.uniform(100, 500, size=50).astype(np.float32))
#     f.run("mean", data = data, realize=False)
# 
#     # pl.DataFrame - multi asset
#     data = pl.DataFrame({ticker: np.random.uniform(100, 500, size=50).astype(np.float32) 
#                 for ticker in ["AAPL","GOOGL","MSFT","AMZN","TSLA"]})
#     f.run("mean", data = data, realize=False)
# 

    # f.run("mean", data = data, realize=False)
