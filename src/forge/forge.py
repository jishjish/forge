import os
# import pynvml
from models import GPU
from dotenv import load_dotenv
from constants.device import NVIDIA_CHIPS
from device.device_info import get_device_type, get_device_info

load_dotenv()
ENV = os.getenv("APP_SETTINGS", "testing")

class Forge:
    def __init__(self):
        # device info
        self.gpu_type = get_device_type()
        self.gpu_info: GPU | None = None
        # self.graph = GraphBuilder(self.ast)
        # self.codegen = CUDACodegen(self.graph)

    def _device_info(self):
        if ENV == 'production':
            get_device_info(self.gpu_type)
        else:
            chip = NVIDIA_CHIPS.get('Tesla T4')
            self.gpu_info = GPU(device_type="CUDA", **chip)
    
    def matmul(self):
        pass
            



if __name__ == "__main__":
    f = Forge()
    f._device_info()
