"""
Function to get device information for NVIDIA chips through use 
of ctypes. Reference via chips dictionary in src/constants/device.py.
"""
import ctypes
from forge.models import NvidiaGPU
from ..constants.device import CHIPS, CUDA_ATTRIBUTES
from ..helpers import DEBUG

class _NvidiaOps:
    def __init__(self):
        self.lib = ctypes.CDLL(CHIPS.get('CUDA')[0])
        try: self.lib.cuInit(0)
        except Exception as e: raise RuntimeError(f"Error initializing CUDA: {e}")
        self.handle = self._get_handle()
        self.model = NvidiaGPU()

    def _get_handle(self):
        handle = ctypes.c_int()
        self.lib.cuDeviceGet(ctypes.byref(handle), 0)
        return handle

    def _get_version(self):
        version = ctypes.c_int()
        self.lib.cuDriverGetVersion(ctypes.byref(version))
        return version

    def _device_info(self):
        version = self._get_version()
        attributes = {}
        for name, attr_id in CUDA_ATTRIBUTES.items():
            result = ctypes.c_int()
            self.lib.cuDeviceGetAttribute(ctypes.byref(result), attr_id, self.handle)
            attributes[name] = result.value 
        return NvidiaGPU(version=version.value, **attributes)



if __name__ == "__main__":
    # c = _NvidiaOps()
    # c._device_info()
    for val in CUDA_ATTRIBUTES:
        print(val)

