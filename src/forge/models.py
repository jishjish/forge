from pydantic import BaseModel


class GPU(BaseModel):
    device_type: str
    cuda_cores: int
    streaming_multiprocessors: int
    threads_per_block: int
    warp_size: int
    max_threads_per_sm: int
    max_threads_total: int