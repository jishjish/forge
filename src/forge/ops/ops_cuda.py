"""
Function to get device information for NVIDIA chips through use 
of ctypes. Reference via chips dictionary in src/constants/device.py.
"""
import ctypes
from forge.constants.device import CHIPS, CUDA_ATTRIBUTES


class _CudaOps:
    def __init__(self):
        self.lib = ctypes.CDLL(CHIPS.get('CUDA')[0])
        self.lib.cuInit(0)
        self.handle = self._get_handle()

    def _get_handle(self):
        handle = ctypes.c_int()
        self.lib.cuDeviceGet(ctypes.byref(handle), 0)
        return handle

    def _get_version(self):
        version = ctypes.c_int()
        self.lib.cuDriverGetVersion(ctypes.byref(version))
        return version

    def _device_info(self):
        for val in CUDA_ATTRIBUTES:
            print(val)



if __name__ == "__main__":
    c = _CudaOps()
    c._device_info()


    
