![Forge Logo](assets/forge_logo.png)
# Forge

A runtime GPU compiler that generates optimized compute kernels for portfolio analytics workloads.

## Overview

Forge queries your GPU's hardware capabilities at runtime and uses that information to generate and compile optimized kernel files. Currently building for CUDA/Metal, with multi-backend support planned.

## Architecture

![Forge Architecture](./assets/forge_architecture.svg)

- **Device Discovery**: Plugin-style system for runtime GPU detection and hardware querying
- **Device State**: Pydantic models representing hardware capabilities and configuration
- **Kernel Generation**: Capability-aware compute kernel emission
- **Compilation**: Clean pipeline from generated source to compiled output

The design enforces dynamic chip identification, with strict separation between device detection, kernel generation, and compilation stages.

## Backends

| Backend | Status |
|--------|--------|
| CUDA | In progress |
| ROCm (AMD) | In progress |
| Metal (Apple) | Planned |

## Status

Early development. Testing on Metal - need cloud GPU for CUDA.

## Roadmap
- End-to-end codegen pipeline with a working `log returns` kernel
<!-- - Tiled shared memory optimization -->
- Additional ops (`correlation matrix`, `efficient frontier`, etc.)
<!-- - Lightweight computation graph for kernel fusion -->
- Multi-backend support (ROCm, Metal)

## Requirements

- Python 3.x
- A supported GPU and its corresponding toolkit


## Getting Started

```bash
git clone https://github.com/jishjish/forge
cd forge
uv pip install -r requirements.txt
# coming soon: forge run examples/log_returns.py
```