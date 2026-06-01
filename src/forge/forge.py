import os
import uuid
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
        self._results = {}

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
    
    def run(self, op: str, realize: bool = False, **kwargs):
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
        state, data_length, output_data_shape = comp.generate(realize=realize, **kwargs) 
        handle = str(uuid.uuid4())
        self._results[handle] = {"data_ptr": state, "data_length": data_length, "output_data_shape": output_data_shape}
        return handle
    
    def realize(self, handle):
        data_ptr = self._results[handle]["data_ptr"]
        output_data_shape = self._results[handle]["output_data_shape"]
        # call realize() through ops to ffi
        _ops_file = [file for file in (Path(__file__).parent/"ops").iterdir() if file.stem.startswith("ops_") and file.stem[len("ops_"):].upper() == self.gpu_info.device_type.upper()]
        module = importlib.import_module(f"forge.ops.{_ops_file[0].stem}")
        cls = getattr(module, f"_{self.gpu_info.device_type.capitalize()}Compile")
        req = cls()
        return req.realize(data_ptr, output_data_shape)
       

if __name__ == "__main__":
    f = Forge()
    import numpy as np
    data=np.random.uniform(100, 500, size=(5000, 3)).astype(np.float32)
    # a = f.run("log_returns", data = data, realize=False)
    test = f.run("mean", data = data, realize=False)
    # print(f.realize(a))
    print(f.realize(test))
