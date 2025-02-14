#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define BLOCK_SIZE 256
// Increase dataset size.
#define N (1 << 26)  // 67,108,864 elements

// Utility: warp-level reduction using shuffle.
__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Kernel: block-level reduction using warp shuffles.
__global__ void sumReduction(int *input, int *output, int n) {
    // Each block uses a fixed BLOCK_SIZE.
    __shared__ int sharedMem[BLOCK_SIZE / 32];  // one element per warp
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int lane = tid % 32;    // index within warp
    int warpId = tid / 32;  // warp index within block

    int sum = (index < n) ? input[index] : 0;
    sum = warpReduceSum(sum);

    if (lane == 0) {
        sharedMem[warpId] = sum;
    }
    __syncthreads();

    if (warpId == 0) {
        int numWarps = BLOCK_SIZE / 32;
        sum = (tid < numWarps) ? sharedMem[lane] : 0;
        sum = warpReduceSum(sum);
    }
    if (tid == 0) {
        output[blockIdx.x] = sum;
    }
}

// Kernel: final reduction using warp shuffles with dynamic shared memory.
__global__ void finalReductionWarp(int *input, int *output, int n) {
    extern __shared__ int sharedMem[]; // one int per active warp
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int lane = tid % 32;
    int warpId = tid / 32;

    int sum = (index < n) ? input[index] : 0;
    sum = warpReduceSum(sum);

    if (lane == 0) {
        sharedMem[warpId] = sum;
    }
    __syncthreads();

    int numWarps = (blockDim.x + 31) / 32;
    if (warpId == 0) {
        sum = (tid < numWarps) ? sharedMem[lane] : 0;
        sum = warpReduceSum(sum);
    }
    if (tid == 0) {
        output[blockIdx.x] = sum;
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

    // First kernel: block-level reduction using warp shuffles.
    sumReduction<<<numBlocks, blockSize>>>(d_input, d_partialSums, N);

    // Iteratively reduce the partial sums using the warp-based final reduction kernel.
    int n = numBlocks;
    int *d_in = d_partialSums;
    int *d_out;
    cudaMalloc(&d_out, numBlocks * sizeof(int));
    while(n > 1) {
        int threads = (n < blockSize) ? nextPow2(n) : blockSize;
        int blocks = (n + threads - 1) / threads;
        // Compute dynamic shared memory size: one int per active warp.
        int sharedMemSize = ((threads + 31) / 32) * sizeof(int);
        finalReductionWarp<<<blocks, threads, sharedMemSize>>>(d_in, d_out, n);
        n = blocks;
        int *temp = d_in;
        d_in = d_out;
        d_out = temp;
    }

    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);
    float elapsedTime_total;
    cudaEventElapsedTime(&elapsedTime_total, start_total, stop_total);

    int result;
    cudaMemcpy(&result, d_in, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Total elapsed time for warpShuffleReduction: %f ms\n", elapsedTime_total);
    printf("Total Sum: %d (expected %d)\n", result, N);

    // Cleanup.
    cudaFree(d_input);
    cudaFree(d_partialSums);
    cudaFree(d_out);
    free(h_input);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    return 0;
}
