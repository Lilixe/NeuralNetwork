#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t e = (call); \
        if (e != cudaSuccess) { \
            printf("[CUDA ERROR] %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            return 1; \
        } \
    } while(0)

__global__ void hello(int* out) {
    out[threadIdx.x] = threadIdx.x + 1;  // +1 so thread 0 writes 1, not 0
}

int main() {
    const int N = 8;
    int h_out[N];
    for (int i = 0; i < N; i++) h_out[i] = -1;  // sentinel: -1 means not written

    int* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));
    printf("d_out allocated at %p\n", (void*)d_out);

    hello<<<1, N>>>(d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_out);

    for (int i = 0; i < N; i++)
        printf("Thread %d wrote: %d\n", i, h_out[i]);

    return 0;
}
