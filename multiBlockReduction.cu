#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define BLOCK_SIZE 256  // Threads per block

// First kernel: Block-wise reduction using shared memory.
__global__ void sumReduction(int *input, int *output, int N) {
    __shared__ int sharedMem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    // Load elements from global memory into shared memory.
    sharedMem[tid] = (index < N) ? input[index] : 0;
    __syncthreads();

    // Parallel reduction in shared memory.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        __syncthreads();
    }

    // Write block sum to global memory.
    if (tid == 0) {
        output[blockIdx.x] = sharedMem[0];
    }
}

// Final reduction kernel (same style as above) that reduces an array of N elements.
__global__ void finalReduction(int *input, int *output, int N) {
    __shared__ int sharedMem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    
    int value = (index < N) ? input[index] : 0;
    sharedMem[tid] = value;
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

// Helper function to compute the next power of 2 (for setting thread count)
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
    // d_partialSums will hold the first reduction (and then subsequent reductions)
    cudaMalloc(&d_partialSums, numBlocks * sizeof(int));

    // Copy data to GPU.
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    float elapsedTime1, elapsedTime2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch first kernel and time it.
    cudaEventRecord(start);
    sumReduction<<<numBlocks, blockSize>>>(d_input, d_partialSums, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime1, start, stop);
    printf("Time elapsed for sumReduction kernel: %f ms\n", elapsedTime1);

    // Iteratively reduce partial sums until only one element remains.
    int n = numBlocks; // Current number of elements in d_partialSums
    int *d_in = d_partialSums;
    int *d_out;
    cudaMalloc(&d_out, numBlocks * sizeof(int)); // Temporary space for reduction

    while(n > 1) {
        // Use fewer threads if n is small.
        int threads = (n < BLOCK_SIZE) ? nextPow2(n) : BLOCK_SIZE;
        int blocks = (n + threads - 1) / threads;
        cudaEventRecord(start);
        finalReduction<<<blocks, threads>>>(d_in, d_out, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime2, start, stop);
        printf("Intermediate finalReduction time: %f ms (blocks: %d, threads: %d, n: %d)\n", 
                elapsedTime2, blocks, threads, n);
        n = blocks;
        // Swap pointers for next iteration.
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
