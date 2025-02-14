#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define BLOCK_SIZE 256
// Increase dataset size.
#define N (1 << 26)  // 67,108,864 elements

// Kernel: block-level reduction using shared memory.
__global__ void sumReduction(int *input, int *output, int n) {
    __shared__ int sharedMem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    sharedMem[tid] = (index < n) ? input[index] : 0;
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

// Kernel: final reduction kernel (shared memory version).
__global__ void finalReduction(int *input, int *output, int n) {
    __shared__ int sharedMem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int index = tid;  // each block processes one “row” of the reduction.
    sharedMem[tid] = (index < n) ? input[index] : 0;
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

// Helper: next power of 2.
int nextPow2(int x) {
    int power = 1;
    while (power < x) {
         power *= 2;
    }
    return power;
}

int main(){
    int blockSize = BLOCK_SIZE;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Allocate and initialize host memory.
    int *h_input = (int*) malloc(N * sizeof(int));
    int *h_result = (int*) malloc(sizeof(int));
    for (int i = 0; i < N; i++){
        h_input[i] = 1;
    }

    // Allocate device memory.
    int *d_input, *d_partialSums;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_partialSums, numBlocks * sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Create overall timing events.
    cudaEvent_t start_total, stop_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventRecord(start_total);

    // First kernel: reduce input to one partial sum per block.
    sumReduction<<<numBlocks, blockSize>>>(d_input, d_partialSums, N);

    // Iteratively reduce the partial sums until one value remains.
    int n = numBlocks;
    int *d_in = d_partialSums;
    int *d_out;
    cudaMalloc(&d_out, numBlocks * sizeof(int));
    while(n > 1) {
        int threads = (n < blockSize) ? nextPow2(n) : blockSize;
        int blocks = (n + threads - 1) / threads;
        finalReduction<<<blocks, threads>>>(d_in, d_out, n);
        n = blocks;
        int *temp = d_in;
        d_in = d_out;
        d_out = temp;
    }

    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);
    float elapsedTime_total;
    cudaEventElapsedTime(&elapsedTime_total, start_total, stop_total);

    // Copy the final result back.
    int result;
    cudaMemcpy(&result, d_in, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Total elapsed time for multiBlockReduction: %f ms\n", elapsedTime_total);
    printf("Total Sum: %d (expected %d)\n", result, N);

    // Cleanup.
    cudaFree(d_input);
    cudaFree(d_partialSums);
    cudaFree(d_out);
    free(h_input);
    free(h_result);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    return 0;
}
