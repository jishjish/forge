 
def generate_metal_log_returns_kernel(**kwargs) -> str:
    s = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void log_returns(
        device const float* prices  [[buffer(0)]],
        device float* returns       [[buffer(1)]],
        constant int& data_length   [[buffer(2)]],
        uint id                     [[thread_position_in_grid]]
    ) {
    """
    return s

def generate_metal_mean_kernel(**kwargs) -> str:
    s = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void mean(
        device const float* returns  [[buffer(0)]],
        device float* averages       [[buffer(1)]],
        constant int& data_length    [[buffer(2)]],
        uint id                      [[thread_position_in_grid]]
    ) {
    """
    return s