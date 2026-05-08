import ctypes
from pathlib import Path
from ..helpers import DEBUG
from forge.models import MetalGPU
from ..constants.device import CHIPS, CUDA_ATTRIBUTES

class _MetalOps:
    def __init__(self):
        self.lib = ctypes.CDLL(CHIPS.get('Metal')[0])
        try: self.lib.cuInit(0)
        except Exception as e: raise RuntimeError(f"Error initializing CUDA: {e}")
        self.handle = self._get_handle()
        self.model = MetalGPU()

# 
#     def _get_version(self):
#         version = ctypes.c_int()
#         self.lib.cuDriverGetVersion(ctypes.byref(version))
#         if DEBUG >= 1: 
#             print(f"[dim]forge ({Path(__file__).name})[/dim] | CUDA Version: {version}")
#         return version

    def _device_info(self):
        version = self._get_version()
        attributes = {}
        for name, attr_id in CUDA_ATTRIBUTES.items():
            result = ctypes.c_int()
            self.lib.cuDeviceGetAttribute(ctypes.byref(result), attr_id, self.handle)
            attributes[name] = result.value 
        if DEBUG >= 1: 
            print(f"[dim]forge ({Path(__file__).name})[/dim] | CUDA Specs: {attributes}")
        return MetalGPU(version=version.value, **attributes)



if __name__ == "__main__":
    c = _MetalOps()

    # c._device_info()
    # for val in CUDA_ATTRIBUTES:
    #     print(val)

