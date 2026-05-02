from models import GPU

def matmul_kernel(gpu: GPU):
    return f"""
__global__ void matmul(float* A, float* B, float* C, int vectorLength)
{{
    int row = 
}}
"""
