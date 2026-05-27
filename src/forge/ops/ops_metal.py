"""
# NOTE: Metal has to be accessed through a C++ shim, because 
Apple provides c++ headers (not a system library). So rather than 
accessing through `CHIPS.get('Metal'), this is invoked through 
`third_party/metal_shim.cpp`. The metal shim file houses 
corresponding functions needed to access device specs.
"""
import numpy as np
import ctypes
from rich import print
from pathlib import Path
from ..helpers import DEBUG
from ..models import MetalGPU
from ..constants.device import CHIPS, METAL_ATTRIBUTES, METAL_GPU_FAMILY

class KernelSpecs(ctypes.Structure):
    _fields_ = [
        ("maxThreadgroupMemoryLength", ctypes.c_int),
        ("maxThreadsPerThreadgroupX", ctypes.c_int),
        ("maxThreadsPerThreadgroupY", ctypes.c_int),
        ("maxThreadsPerThreadgroupZ", ctypes.c_int),
        ("recommendedMaxWorkingSetSize", ctypes.c_int)
    ]

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
        for name, (func, res_type) in METAL_ATTRIBUTES.items():
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
            fn.restype = getattr(ctypes, res_type)
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
    
    def _compile(
        self, 
        source_code, 
        function_name, 
        gpu,
        data,
        realize
    ):
        # get function to compile source code
        fn = getattr(self.lib, "forge_compile_source")
        # set arg and return types for ffi
        fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        fn.restype = ctypes.c_void_p
        result = fn(self.device, source_code.encode("utf-8"), function_name.encode("utf-8"))
        
        # take lib and generate pipeline (follow similar arg/return types)
        pipe_fn = getattr(self.lib, "forge_generate_pipeline")
        pipe_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        pipe_fn.restype = ctypes.c_void_p

        # pipeline result returns void, have to unwrap `.value` (memory address)
        pipe_result = ctypes.c_void_p(pipe_fn(self.device, ctypes.c_void_p(result))).value

        dispatch_fn = getattr(self.lib, "forge_dispatch_pipeline") 
        data_ptr = data.ctypes.data
        byte_length = data.nbytes

        extracted_specs = [key for key in gpu.model_fields.keys() if key in [f[0] for f in KernelSpecs._fields_]]
        kernel_specs = KernelSpecs(**{k: getattr(gpu, k) for k in extracted_specs})
        dispatch_fn.argtypes = [
            ctypes.c_void_p,                # device
            ctypes.c_void_p,                # pipeline 
            ctypes.POINTER(KernelSpecs),    # kernel_specs
            ctypes.c_int,                   # data_length 
            ctypes.c_void_p,                # data_ptr
            ctypes.c_int                    # byte_length
        ]
        dispatch_fn.restype = ctypes.c_void_p
        dispatch_res = dispatch_fn(self.device, pipe_result, kernel_specs, len(data), data_ptr, byte_length)
        if realize:
            read_fn = getattr(self.lib, "forge_read_output_buf")
            read_fn.argtypes = ctypes.c_void_p, ctypes.c_int
            # read_fn.restype = ctypes.c_void_p
            read_fn.restype = ctypes.POINTER(ctypes.c_float)
            # data_length = len(data)
            data_length = data.size     # pass size (not len) so we can retrieve 2d array data
            data = read_fn(dispatch_res, data_length)
            return np.ctypeslib.as_array(data, shape=(data_length,)).copy()
        else:
            return dispatch_res
    

if __name__ == "__main__":
    c = _MetalOps()
    print(c._device_info())