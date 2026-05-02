from pydantic import BaseModel

class CudaGPU(BaseModel):
    version: str
    arch: str                       # f string "fm_{major}{minor}" to be used for compiler reference
    max_threads_per_block: int
    max_block_dim_x: int
    max_block_dim_y: int
    max_block_dim_z: int
    max_grid_dim_x: int
    max_grid_dim_y: int
    max_grid_dim_z: int
    warp_size: int
    sm_count: int
    max_threads_per_sm: int
