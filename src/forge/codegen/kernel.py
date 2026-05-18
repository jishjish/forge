

def generate_metal_kernel(**kwargs) -> str:
    return """
    #include <metal_stdlib>
    using namespace metal;

    kernel void log_returns(
        device const float* prices [[buffer(0)]],
        device float* returns      [[buffer(1)]],
        uint id [[thread_position_in_grid]]
    ) {
    """