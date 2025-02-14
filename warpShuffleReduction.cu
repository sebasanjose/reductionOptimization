#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define BLOCK_SIZE 256  // Threads per block

// Utility function: Warp shuffle for intra-warp reduction.
__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// First kernel: Block-wise reduction using warp shuffles.
__global__ void sumReduction(int *input, int *output, int N) {
    __shared__ int sharedMem[BLOCK_SIZE / 32];  // For inter-warp communication

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int lane = tid % 32;    // Index within the warp.
    int warpId = tid / 32;  // Which warp within the block.

    // Load element into register.
    int sum = (index < N) ? input[index] : 0;

    // Intra-warp reduction using warp shuffle.
    sum = warpReduceSum(sum);

    // Write reduced value of each warp to shared memory.
    if (lane == 0) sharedMem[warpId] = sum;
    __syncthreads();

    // Final reduction within the first warp.
    if (warpId == 0) {
        sum = (tid < (BLOCK_SIZE / 32)) ? sharedMem[lane] : 0;
        sum = warpReduceSum(sum);
    }

    // Write the block's final sum.
    if (tid == 0) {
        output[blockIdx.x] = sum;
    }
}

// Final reduction kernel using warp shuffles. This kernel reduces an array of N elements.
__global__ void finalReductionWarp(int *input, int *output, int N) {
    __shared__ int sharedMem[BLOCK_SIZE / 32];  // For inter-warp communication

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int lane = tid % 32;
    int warpId = tid / 32;

    int sum = (index < N) ? input[index] : 0;
    sum = warpReduceSum(sum);
    if(lane == 0) {
        sharedMem[warpId] = sum;
    }
    __syncthreads();

    if(warpId == 0) {
        sum = (tid < (blockDim.x / 32)) ? sharedMem[lane] : 0;
        sum = warpReduceSum(sum);
    }
    if(tid == 0) {
        output[blockIdx.x] = sum;
    }
}

// Helper function to compute the next power of 2.
int nextPow2(int x) {
    int power = 1;
    while (power < x) {
        power *= 2;
    }
    return power;
}

int main() {
    int N = 1 << 20;  // 1,048,576 elements
    int blockSize = BLOCK_SIZE;
    int numBlocks = (N + blockSize - 1) / blockSize;

    int *h_input, *h_result;
    int *d_input, *d_partialSums;
    h_input = (int*) malloc(N * sizeof(int));
    h_result = (int*) malloc(sizeof(int));

    // Initialize input array (all ones).
    for (int i = 0; i < N; i++) {
        h_input[i] = 1;
    }

    // Allocate GPU memory.
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_partialSums, numBlocks * sizeof(int));

    // Copy data to GPU.
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    float elapsedTime1, elapsedTime2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch the first kernel (block-wise reduction using warp shuffles) and time it.
    cudaEventRecord(start);
    sumReduction<<<numBlocks, blockSize>>>(d_input, d_partialSums, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime1, start, stop);
    printf("Time elapsed for sumReduction kernel (warp shuffle): %f ms\n", elapsedTime1);

    // Iteratively reduce partial sums using the warp-based final reduction kernel.
    int n = numBlocks;
    int *d_in = d_partialSums;
    int *d_out;
    cudaMalloc(&d_out, numBlocks * sizeof(int)); // Temporary space

    while(n > 1) {
        int threads = (n < BLOCK_SIZE) ? nextPow2(n) : BLOCK_SIZE;
        int blocks = (n + threads - 1) / threads;
        cudaEventRecord(start);
        finalReductionWarp<<<blocks, threads>>>(d_in, d_out, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime2, start, stop);
        printf("Intermediate finalReduction (warp shuffle) time: %f ms (blocks: %d, threads: %d, n: %d)\n",
                elapsedTime2, blocks, threads, n);
        n = blocks;
        int *temp = d_in;
        d_in = d_out;
        d_out = temp;
    }

    // Copy the final result back to host.
    cudaMemcpy(h_result, d_in, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Total Sum: %d (expected %d)\n", *h_result, N);

    // Cleanup.
    cudaFree(d_input);
    cudaFree(d_partialSums);
    cudaFree(d_out);
    free(h_input);
    free(h_result);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
