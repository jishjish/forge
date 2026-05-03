CHIPS = {
    "CUDA": ("libcuda.so", "cuInit(0)"),
    "AMD": ("libamdhip64.so", "hipInit(0)"),
    "Metal": ("libMetal.dylib", "MTLCreateSystemDefaultDevice")
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