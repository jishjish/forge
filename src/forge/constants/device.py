SUPPORTED_DEVICES = ["CUDA", "AMD", "ARM64"]

NVIDIA_CHIPS = {
    "Tesla T4": {
        "cuda_cores": 2_560,
        "streaming_multiprocessors": 40,
        "threads_per_block": 1_024,      # hardware max
        "warp_size": 32,
        "max_threads_per_sm": 1_024,     # Turing architecture
        "max_threads_total": 40_960,     # 40 SM * 1024
    }
}