"""
# NOTE: Metal has to be accessed through a C++ shim, because 
Apple provides c++ headers (not a system library). So rather than 
accessing through `CHIPS.get('Metal'), this is invoked through 
`third_party/metal_shim.cpp`.

Because of this we reference the build path and then the function call to instantiate. 
"""


# ******************* SUPPORTED CHIPS *******************
CHIPS = {
    "CUDA": ("libcuda.so", "cuInit(0)"),
    "AMD": ("libamdhip64.so", "hipInit(0)"),
    "Metal": ("third_party/build/libforge_metal.dylib", "forge_get_device")
}

COMPILE_COMMANDS = {
    "Metal": ["xcrun", "-sdk", "macosx", "metal", "-c", "{src}", "-o", "{out}"],
    "CUDA": ["nvcc", "-arch={arch}", "-o", "{out}", "{src}"],
}


# ******************* CUDA ATTRIBUTES *******************
CUDA_ATTRIBUTES = {
    "compute_capability_major": 75,
    "compute_capability_minor": 76,
    "max_threads_per_block": 1,
    "max_block_dim_x": 2,
    "max_block_dim_y": 3,
    "max_block_dim_z": 4,
    "max_grid_dim_x": 5,
    "max_grid_dim_y": 6,
    "max_grid_dim_z": 7,
    "warp_size": 10,
    "sm_count": 16,
    "max_threads_per_sm": 39,
}



# ******************* METAL ATTRIBUTES *******************
# METAL_ATTRIBUTES = {
#     "name": {"forge_device_name", "ctypes.c_char_p"}, 
#     "maxThreadgroupMemoryLength": {"forge_max_threadgroup_memory", "ctypes.c_int"}, 
#     "maxThreadsPerThreadgroupX": {"forge_max_threads_per_group_x", "ctypes.c_int"},
#     "maxThreadsPerThreadgroupY": "forge_max_threads_per_group_y", "ctypes.c_int"},
#     "maxThreadsPerThreadgroupZ": {"forge_max_threads_per_group_z", "ctypes.c_int"},
#     "recommendedMaxWorkingSetSize": {"forge_recommended_max_working_size", "ctypes.c_unit64"},
#     "supportsFamily": {"forge_supports_family", "ctypes.c_char_p"}
# }

METAL_ATTRIBUTES = {
    "name":                        ("forge_device_name",                  "c_char_p"),
    "maxThreadgroupMemoryLength":  ("forge_max_threadgroup_memory",       "c_uint64"),
    "maxThreadsPerThreadgroupX":   ("forge_max_threads_per_group_x",      "c_uint64"),
    "maxThreadsPerThreadgroupY":   ("forge_max_threads_per_group_y",      "c_uint64"),
    "maxThreadsPerThreadgroupZ":   ("forge_max_threads_per_group_z",      "c_uint64"),
    "recommendedMaxWorkingSetSize":("forge_recommended_max_working_size", "c_uint64"),
    "supportsFamily":              ("forge_supports_family",              "c_uint64"),
}

# ******************* METAL GPU FAMILY *******************
METAL_GPU_FAMILY = {
    1001: "apple1",  # A7
    1002: "apple2",  # A8
    1003: "apple3",  # A9/A10
    1004: "apple4",  # A11
    1005: "apple5",  # A12
    1006: "apple6",  # A13
    1007: "apple7",  # A14/M1
    1008: "apple8",  # A15/A16/M2
    1009: "apple9",  # A17/M3/M4
}