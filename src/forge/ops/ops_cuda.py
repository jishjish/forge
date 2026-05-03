"""
Function to get device information for NVIDIA chips through use 
of ctypes. Reference via chips dictionary in src/constants/device.py.
"""
import ctypes
from forge.constants.device import CHIPS, CUDA_ATTRIBUTES

class _NvidiaOps:
    def __init__(self):
        self.lib = ctypes.CDLL(CHIPS.get('CUDA')[0])
        self.lib.cuInit(0)
        self.handle = self._get_handle()

        self.model = NvidiaGPU

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
    # c = _NvidiaOps()
    # c._device_info()
    for val in CUDA_ATTRIBUTES:
        print(val)




    # class NvidiaGPU(BaseModel):
    #     version: str = "unknown"
    #     arch: str = "sm_75"                    # f string "fm_{major}{minor}" to be used for compiler reference
    #     max_threads_per_block: int = 1_024
    #     max_block_dim_x: int = 1_024
    #     max_block_dim_y: int = 1_024
    #     max_block_dim_z: int = 64
    #     max_grid_dim_x: int = 2_147_483_647
    #     max_grid_dim_y: int = 65_535
    #     max_grid_dim_z: int = 65_535
    #     warp_size: int = 32
    #     sm_count: int = 40
    #     max_threads_per_sm: int = 1_024