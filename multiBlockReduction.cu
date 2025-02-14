#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define BLOCK_SIZE 256  // Threads per block

// Kernel: block-level reduction using shared memory.
__global__ void sumReduction(int *input, int *output, int N) {
    __shared__ int sharedMem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    sharedMem[tid] = (index < N) ? input[index] : 0;
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

// Kernel: final reduction kernel (same idea as above).
__global__ void finalReduction(int *input, int *output, int N) {
    __shared__ int sharedMem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int index = tid;  // Only one block is expected to launch this kernel.
    sharedMem[tid] = (index < N) ? input[index] : 0;
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
    int N = 1 << 20;  // 1,048,576 elements
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
        int threads = (n < BLOCK_SIZE) ? nextPow2(n) : BLOCK_SIZE;
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
    cudaMemcpy(h_result, d_in, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Total elapsed time for multiBlockReduction: %f ms\n", elapsedTime_total);
    printf("Total Sum: %d (expected %d)\n", *h_result, N);

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
