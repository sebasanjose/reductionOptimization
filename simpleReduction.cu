#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Simple parallel reduction kernel using shared memory.
__global__ void simpleReductionKernel(int *input, int *output, int N) {
    __shared__ int sharedMem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    // Load data from global memory into shared memory.
    sharedMem[tid] = (i < N) ? input[i] : 0;
    __syncthreads();

    // Perform reduction in shared memory.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if(tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        __syncthreads();
    }
    
    // Write the block's sum to global memory.
    if(tid == 0) {
        output[blockIdx.x] = sharedMem[0];
    }
}

int main(){
    int N = 1 << 20; // 1 Million elements
    int size = N * sizeof(int);
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    int *h_input = (int*) malloc(size);
    int *h_output = (int*) malloc(numBlocks * sizeof(int));

    // Initialize input array (all ones).
    for (int i = 0; i < N; i++) {
        h_input[i] = 1;
    }

    int *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, numBlocks * sizeof(int));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel for block-level reduction.
    simpleReductionKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time elapsed for simpleReduction kernel: %f ms\n", elapsedTime);

    // Copy partial results back to host and perform final reduction on CPU.
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    int total = 0;
    for (int i = 0; i < numBlocks; i++){
        total += h_output[i];
    }
    printf("Total Sum: %d\n", total);

    // Cleanup.
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
