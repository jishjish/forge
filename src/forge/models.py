from pydantic import BaseModel

class NvidiaGPU(BaseModel):
    version: int = 0
    arch: str = "sm_75"                    # f string "fm_{major}{minor}" to be used for compiler reference
    max_threads_per_block: int = 1_024
    max_block_dim_x: int = 1_024
    max_block_dim_y: int = 1_024
    max_block_dim_z: int = 64
    max_grid_dim_x: int = 2_147_483_647
    max_grid_dim_y: int = 65_535
    max_grid_dim_z: int = 65_535
    warp_size: int = 32
    sm_count: int = 40
    max_threads_per_sm: int = 1_024

# TODO: update for amd specs
class AmdGPU(BaseModel):
    version: str
    arch: str                       
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


# TODO: update for apple specs
class MetalGPU(BaseModel):
    version: str
    # arch: str                       
    max_threads_per_block: int
    # max_block_dim_x: int
    # max_block_dim_y: int
    # max_block_dim_z: int
    # max_grid_dim_x: int
    # max_grid_dim_y: int
    # max_grid_dim_z: int
    # warp_size: int
    # sm_count: int
    # max_threads_per_sm: int



