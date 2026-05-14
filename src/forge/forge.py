import os
import inspect
import importlib
from rich import print
from pathlib import Path
from dotenv import load_dotenv
import models as _models
from .device.device import Device
from .helpers import DEBUG

load_dotenv()
ENV = os.getenv("APP_SETTINGS", "testing")

class Forge:
    _gpu_models = [c[1] for c in inspect.getmembers(_models, inspect.isclass) if c[0] != 'BaseModel']
    _portfolio_ops = [file.stem[len("op_"):] for file in (Path(__file__).parent/"codegen/portfolio").iterdir() if file.stem.startswith("op")]
    _linalg_ops = [file.stem[len("op_"):] for file in (Path(__file__).parent/"codegen/linalg").iterdir() if file.stem.startswith("op")]

    def __init__(self):
        self.device = Device()
        self.gpu_info = None
        # self.graph = GraphBuilder(self.ast)
        # self.codegen = CUDACodegen(self.graph)

    @classmethod
    def supported_gpus(cls): return [m.__name__ for m in cls._gpu_models]

    @classmethod
    def supported_ops(cls): return {"portfolio_ops": cls._portfolio_ops, "linalg_ops": cls._linalg_ops}

    def _device_info(self):
        # returns corresponding pydantic GPU model; (Forge --> Device --> get_device_info() --> gpu ops)
        if ENV == 'production': 
            gpu_info = self.device.get_device_info()
            assert isinstance(gpu_info, tuple(self._gpu_models)), f"Unsupported GPU: {type(gpu_info).__name__}. Supported: {[c.__name__ for c in self._gpu_models]}"
            self.gpu_info = gpu_info
            if DEBUG >= 1: 
                print(f"[dim]forge ({Path(__file__).name})[/dim] | [dim]Device initialized:[/dim] {self.gpu_info}")
        else: 
            self.gpu_info = _models.NvidiaGPU()
            if DEBUG >= 1: 
                print(f"[dim]forge ({Path(__file__).name})[/dim] | [white]env set to testing, returning base NVIDIA model[/white]")
    
    def run(self, op: str, **kwargs):
        assert op in self._portfolio_ops or op in self._linalg_ops, \
            f"{op} not found. Supported ops: {self._portfolio_ops + self._linalg_ops}"

        # import bases on provided op; either through `portfolio` or `linalg`
        if op in self._portfolio_ops: import_path = f"src.forge.codegen.portfolio.op_{op}"
        else: import_path = f"src.forge.codegen.linalg.op_{op}"
        if DEBUG >= 1:
            print(f"[dim]forge ({Path(__file__).name})[/dim] | Import path: {import_path} for {op} operation")
            print(f"[dim]forge ({Path(__file__).name})[/dim] | Kwargs: {kwargs}")

        try: ops = importlib.import_module(import_path)
        except ModuleNotFoundError: raise RuntimeError(f"Op module not found: {import_path}")
       

if __name__ == "__main__":
    f = Forge()
    print(f._device_info()) 
    # print(f._gpu_models)
    # print(f.run('matmul'))
    # print(f.supported_ops())
    # print(DEBUG)
