import os
import inspect
import numpy as np
from rich import print
from pathlib import Path
from .helpers import DEBUG
import forge.models as _models
from dotenv import load_dotenv
from .device.device import Device
from forge.codegen.compile import Compile
load_dotenv()

class Forge:
    _gpu_models = [c[1] for c in inspect.getmembers(_models, inspect.isclass) if c[0] != 'BaseModel']
    _portfolio_ops = [file.stem[len("op_"):] for file in (Path(__file__).parent/"codegen/portfolio").iterdir() if file.stem.startswith("op")]
    _linalg_ops = [file.stem[len("op_"):] for file in (Path(__file__).parent/"codegen/linalg").iterdir() if file.stem.startswith("op")]

    def __init__(self):
        self.device = Device()
        self.gpu_info = None
        self._device_info()

    @classmethod
    def supported_gpus(cls): return [m.__name__ for m in cls._gpu_models]

    @classmethod
    def supported_ops(cls): return {"portfolio_ops": cls._portfolio_ops, "linalg_ops": cls._linalg_ops}

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
        # import bases on provided op; either through `portfolio` or `linalg`
        if op in self._portfolio_ops: import_path = f"forge.codegen.portfolio.op_{op}"
        else: import_path = f"forge.codegen.linalg.op_{op}"
        if DEBUG >= 1:
            print(f"[dim]forge ({Path(__file__).name})[/dim] | Import path: {import_path} for {op} operation")
            print(f"[dim]forge ({Path(__file__).name})[/dim] | Kwargs: {kwargs}")
        # start handling of compilations
        assert self.gpu_info is not None, "Device not initialized"
        comp = Compile(self.gpu_info, import_path, op)
        comp.generate(realize=realize, **kwargs) 

if __name__ == "__main__":
    f = Forge()
    data = np.random.uniform(100, 500, size=500_000).astype(np.float32)
    f.run("log_returns", data=data)
