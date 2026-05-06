# from models import GPU

# Basic matmul kernel
# __global__ void matmul(float* A, float* B, float* C, int N)
# {
#     int row = blockIdx.y * blockDim.y + threadIdx.y;
#     int col = blockIdx.x * blockDim.x + threadIdx.x; 
#     # guard against threads falling outside matrix
#     # grid dimensions dont divide evenly into N
#     # some threads get launched but have no valid
#     # work to do
#     if (row < N && col < N) {
#         # float literal 0
#         float sum = 0.0f;
#         for (int k = 0, k < N, k++) {
#             sum += A[row * n + k] * B[k * N + col];
#         }
#         C[row * N + col] = sum;
#     }
# }


def square_matmul_kernel():
    """ base implementation of a matmul on equal length vectors """
    kernel = f"""
    __global__ void matmul(float* A, float* B, float* C, int N)
    {{
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x; 
        if (row < N && col < N) {{
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {{
                sum += A[row * N + k] * B[k * N + col];
            }}
            C[row * N + col] = sum;
        }}
    }}
    """
    return kernel






if __name__ == "__main__":
    print(square_matmul_kernel())