import inspect
from .matmul import square_matmul_kernel
from .memory import unified_memory
import src.forge.models as _models
from pydantic import BaseModel

class Compile:
    # _gpu_models = [c[0] for c in inspect.getmembers(_models, inspect.isclass) if c[0] != 'BaseModel']
    _gpu_models = {name: cls for name, cls in inspect.getmembers(_models, inspect.isclass) 
                   if issubclass(cls, BaseModel) and cls is not BaseModel}

    def __init__(self):
        pass

    def compile_square_matmul(self, N: int, gpu):
        kernel = square_matmul_kernel()                 # the __global__ function
        host = unified_memory()                         # cudamalloc, launch, cudafree
        return kernel + "\n" + host


if __name__ == "__main__":
    c = Compile()
    print(c.compile_square_matmul(N=1024, gpu=_models.NvidiaGPU))
    # print(c._gpu_models)