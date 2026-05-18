"""
# NOTE: Metal has to be accessed through a C++ shim, because 
Apple provides c++ headers (not a system library). So rather than 
accessing through `CHIPS.get('Metal'), this is invoked through 
`third_party/metal_shim.cpp`. The metal shim file houses 
corresponding functions needed to access device specs.
"""

import ctypes
from rich import print
from pathlib import Path
from ..helpers import DEBUG
from ..models import MetalGPU
from ..constants.device import CHIPS, METAL_ATTRIBUTES, METAL_GPU_FAMILY

class _MetalOps:
    def __init__(self):
        # ffi through c++
        self.lib = ctypes.CDLL(CHIPS.get("Metal")[0])
        try:
            # set the result types for the c++ shim calls
            self.lib.forge_get_device.restype = ctypes.c_void_p
            self.lib.forge_device_name.restype = ctypes.c_char_p
            self.device = self.lib.forge_get_device()
        except Exception as e: raise RuntimeError(f"Error initializing Metal: {e}")
        self.model = MetalGPU()

    def _device_info(self):
        attributes = {}
        for name, func in METAL_ATTRIBUTES.items():
            if name == "supportsFamily":
                fn = getattr(self.lib, func)
                fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
                fn.restype = ctypes.c_bool
                # iterate backward (most recent device) 
                for i in range(len(METAL_GPU_FAMILY), 0, -1):
                    if fn(self.device, 1000 + i):
                        family_reference = METAL_GPU_FAMILY.get(1000+i)
                        attributes[name] = family_reference
                        break
                continue
            fn = getattr(self.lib, func)
            fn.argtypes = [ctypes.c_void_p]
            result = fn(self.device)
            attributes[name] = result
        if DEBUG >= 1: 
            print(f"[dim]forge ({Path(__file__).name})[/dim] | Metal Specs: {attributes}")
        return MetalGPU(**attributes)
            

class _MetalCompile:
    def __init__(self):
        # ffi through c++
        self.lib = ctypes.CDLL(CHIPS.get("Metal")[0])
        try:
            # set the result types for the c++ shim calls
            self.lib.forge_get_device.restype = ctypes.c_void_p
            self.device = self.lib.forge_get_device()
        except Exception as e: raise RuntimeError(f"Error initializing Metal: {e}")
    
    def _compile(self, source_code, function_name):
        fn = getattr(self.lib, "forge_compile_source")
        fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        fn.restype = ctypes.c_void_p
        result = fn(self.device, source_code.encode("utf-8"), function_name.encode("utf-8"))
        
        pipe_fn = getattr(self.lib, "forge_generate_pipeline")
        pipe_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        pipe_fn.restype = ctypes.c_void_p

        # pipeline result returns void, have to unwrap
        pipe_result = ctypes.c_void_p(pipe_fn(self.device, ctypes.c_void_p(result)))
        return pipe_result.value
    

if __name__ == "__main__":
    c = _MetalOps()
    c._device_info()