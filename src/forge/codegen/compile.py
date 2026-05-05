from typing import Any
from .matmul import square_matmul_kernel
from .memory import unified_memory
form .

class Compile:
    def __init__(self):
        pass

    def compile_square_matmul(self, N: int, gpu: NvidiaGPU | AmdGPU | MetalGPU]):
        kernel = square_matmul_kernel()     # the __global__ function
        host = unified_memory()             # cudaMalloc, launch, cudaFree
        return kernel + "\n" + host


if __name__ == "__main__":
    c = Compile()
    print(c.compile_square_matmul(N=1024, gpu=NvidiaGPU))