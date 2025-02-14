#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

// Kernel: simple reduction within a block using shared memory.
__global__ void simpleReductionKernel(int *input, int *output, int N) {
    __shared__ int sharedMem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sharedMem[tid] = (i < N) ? input[i] : 0;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sharedMem[0];
    }
}

int main(){
    int N = 1 << 20; // 1,048,576 elements
    int size = N * sizeof(int);
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate and initialize host memory.
    int *h_input = (int*) malloc(size);
    int *h_partialSums = (int*) malloc(numBlocks * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_input[i] = 1;
    }

    // Allocate device memory.
    int *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, numBlocks * sizeof(int));
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Create overall timing events.
    cudaEvent_t start_total, stop_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventRecord(start_total);

    // Launch the kernel.
    simpleReductionKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, N);

    // Copy block-level partial sums back and finish the reduction on CPU.
    cudaMemcpy(h_partialSums, d_output, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    int total = 0;
    for (int i = 0; i < numBlocks; i++){
        total += h_partialSums[i];
    }

    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);
    float elapsedTime_total;
    cudaEventElapsedTime(&elapsedTime_total, start_total, stop_total);
    printf("Total elapsed time for simpleReduction: %f ms\n", elapsedTime_total);
    printf("Total Sum: %d (expected %d)\n", total, N);

    // Cleanup.
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_partialSums);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    return 0;
}
