import os
import inspect
from src.forge.device.device import Device
import src.forge.models as _models
from dotenv import load_dotenv

load_dotenv()
ENV = os.getenv("APP_SETTINGS", "testing")

class Forge:
    _gpu_models = [c[0] for c in inspect.getmembers(_models, inspect.isclass) if c[0] != 'BaseModel']

    def __init__(self):
        self.device = Device()
        self.gpu_info = None
        # self.graph = GraphBuilder(self.ast)
        # self.codegen = CUDACodegen(self.graph)

    def _device_info(self):
        # returns corresponding pydantic GPU model
        # (Forge --> Device --> get_device_info() --> gpu ops)
        if ENV == 'production': self.gpu_info = self.device._init_device()
        else: 
            self.gpu_info = _models.NvidiaGPU()
            print(self.gpu_info)
    
    def matmul(self):
        pass
            

if __name__ == "__main__":
    f = Forge()
    f._device_info()




