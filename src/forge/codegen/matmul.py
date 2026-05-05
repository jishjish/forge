from models import GPU

# Basic matmul kernel
# __global__ void matmul(float* A, float* B, float* C, int N)
# {
#     int row = blockIdx.y * blockDim.y + threadIdx.y;
#     int col = blockIdx.x * blockDim.x + threadIdx.x; 
#     if (row < N && col < N) {
#         float sum = 0.0f;
#         for (int k = 0, k < N, k++) {
#             sum += A[row * n + k] * B[k * N + col];
#         }
#         C[row * N + col] = sum;
#     }
# }
# 

def matmul_kernel():
    kernel = f"""
    __global__ void matmul(float* A, float* B, float* C, int N)
    {{
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x; 
        if (row < N && col < N) {{
            float sum = 0.0f;
            for (int k = 0, k < N, k++) {{
                sum += A[row * n + k] * B[k * N + col];
            }}
            C[row * N + col] = sum;
        }}
    }};
    """


