import os
from models import NvidiaGPU, AmdGPU, MetalGPU
from dotenv import load_dotenv
from forge.device.device import Device

load_dotenv()
ENV = os.getenv("APP_SETTINGS", "testing")

class Forge:
    def __init__(self):
        self.device = Device()
        self.gpu_info: NvidiaGPU | AmdGPU | MetalGPU | None = None
        # self.graph = GraphBuilder(self.ast)
        # self.codegen = CUDACodegen(self.graph)

    def _device_info(self):
        # returns corresponding pydantic GPU model
        # (Forge --> Device --> get_device_info() --> gpu ops)
        if ENV == 'production': self.gpu_info = self.device._init_device()
        else: self.gpu_info = NvidiaGPU()
    
    def matmul(self):
        pass
            



if __name__ == "__main__":
    f = Forge()
    f._device_info()
