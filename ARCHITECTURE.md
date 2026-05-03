# Goal
To convert Python code to CUDA (or other GPU based syntax).

## Workflow
Forge class --> device check --> computation graph --> CUDA / ROCm (AMD)

1. Forge Class 
    - orchestration for device type and chip checks
    - if testing environment, default is NVIDIA chip in models.py
2. Device Class
    - attempt to initialize chips for supported devices
    - identifies `device type` and `device hardware` through device_info.py 


## Project Structure
Forge orchestration class
Graph computation
Codegen





