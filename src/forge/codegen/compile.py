# from .memory import unified_memory
# import forge.models as _models
# import ctypes
# import tempfile
# import traceback
import importlib
# import subprocess
from rich import print
from pathlib import Path
from pydantic import BaseModel
from forge.helpers import DEBUG

from ..constants.device import CHIPS, METAL_ATTRIBUTES, METAL_GPU_FAMILY

class Compile:
    def __init__(self, gpu: BaseModel, import_path: str, op_name):
        self.gpu = gpu
        self.import_path = import_path
        self.op_name = op_name
        self.output_path = Path(__file__).resolve().parent / "source_code" / f"{op_name}{self.gpu.file_ext}"
        if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] | GPU: {self.gpu.device_type}, op: {self.import_path}, out: {self.output_path}")

    def _build_kernel(self, **kwargs) -> str:
        try: kernel_module = importlib.import_module("forge.codegen.kernel")
        except ModuleNotFoundError: raise RuntimeError(f"Kernel module not found: {self.import_path}")
        kernel_cls = getattr(kernel_module, f"generate_{self.gpu.device_type.lower()}_kernel")
        return kernel_cls(**kwargs)

    def _build_ops(self, **kwargs) -> str:
        try: op_module = importlib.import_module(self.import_path)
        except ModuleNotFoundError: raise RuntimeError(f"Op module not found: {self.import_path}")
        op_cls = getattr(op_module, f"generate_{self.gpu.device_type.lower()}")
        return op_cls(self.gpu, **kwargs)

    def _compile(self, source_code):
        _ops_file = [file for file in (Path(__file__).parent.parent/"ops").iterdir() if file.stem.startswith("ops_") and file.stem[len("ops_"):].upper() == self.gpu.device_type.upper()]
        module = importlib.import_module(f"forge.ops.{_ops_file[0].stem}")
        cls = getattr(module, f"_{self.gpu.device_type.capitalize()}Compile")
        req = cls()
        req._compile(source_code, self.op_name)
        if DEBUG >= 1: 
            print(f"[dim]forge ({Path(__file__).name})[/dim] | ops file: {_ops_file}")
            print(f"[dim]forge ({Path(__file__).name})[/dim] | importing module: {module}")
            print(f"[dim]forge ({Path(__file__).name})[/dim] | cls: {cls}")

    def _handle_data(self, gpu: BaseModel, data):
        pass

    def generate(self):
        source_code = self._build_kernel() + self._build_ops() + "\n}"
        # Path(self.output_path).write_text(source_code)
        self._compile(source_code)
        if DEBUG >= 1: print(f"[dim]forge ({Path(__file__).name})[/dim] | Updated source code and wrote to: {self.output_path}")

if __name__ == "__main__":
    # c = Compile()
    # print(c.compile_square_matmul(N=1024, gpu=_models.NvidiaGPU))
    # print(c._gpu_models)
    # script_dir = Path(__file__).resolve().parent / "source_code"
    # print(script_dir)
    pass