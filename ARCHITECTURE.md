# Goal
To convert Python code to CUDA (or other GPU based syntax).

## Workflow
Forge class --> device check --> computation graph --> CUDA / ROCm (AMD)

1. Forge Class 
    - highest level; entry point for device type and device information checks
    - if testing environment, default is NVIDIA chip in models.py
2. Device Class
    - orchestration class for identifying and getting device information
    - identifies `device type` and `device hardware` through device_info.py 
3. Ops Classes (intermediary classes)
    - class for each supported chip (given variance in chip structrue and reporting)
        - ex: `_NvidiaOps` at `forge/ops/ops_cuda.py`
    - [instructions for adding support for new chips](src/forge/device/device_info.py)
    - commonality: Device class calls through `_device_info()`


## Project Structure
Forge orchestration class
Graph computation
Codegen





