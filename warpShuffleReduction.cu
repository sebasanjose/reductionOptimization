#include <stdio.h>

#define BLOCK_SIZE 256  // Threads per block

// Utility function: Warp shuffle for intra-warp reduction
__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// First kernel: Block-wise reduction using warp shuffles
__global__ void sumReduction(int *input, int *output, int N) {
    __shared__ int sharedMem[BLOCK_SIZE / 32];  // Shared memory for inter-warp communication

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    int lane = tid % 32;    // Thread index within warp
    int warpId = tid / 32;  // Warp index within block

    // Load element into register
    int sum = (index < N) ? input[index] : 0;

    // Intra-warp reduction using warp shuffles
    sum = warpReduceSum(sum);

    // Store warp-level sum into shared memory
    if (lane == 0) sharedMem[warpId] = sum;
    __syncthreads();

    // Final reduction using the first warp
    if (warpId == 0) {
        sum = (lane < (BLOCK_SIZE / 32)) ? sharedMem[lane] : 0;
        sum = warpReduceSum(sum);
    }

    // Store block's final sum
    if (tid == 0) {
        output[blockIdx.x] = sum;
    }
}

// Second kernel: Final sum of partial sums
__global__ void finalReduction(int *input, int *output, int N) {
    int sum = (threadIdx.x < N) ? input[threadIdx.x] : 0;
    sum = warpReduceSum(sum);

    if (threadIdx.x == 0) {
        *output = sum;
    }
}

int main() {
    int N = 1 << 20;  // 1 Million elements
    int blockSize = BLOCK_SIZE;
    int numBlocks = (N + blockSize - 1) / blockSize;

    int *h_input, *h_output;
    int *d_input, *d_partialSums, *d_finalSum;

    // Allocate host memory
    h_input = (int*) malloc(N * sizeof(int));
    h_output = (int*) malloc(sizeof(int));

    // Initialize input array
    for (int i = 0; i < N; i++) {
        h_input[i] = 1;  // Example: all elements are 1
    }

    // Allocate GPU memory
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_partialSums, numBlocks * sizeof(int));
    cudaMalloc(&d_finalSum, sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch first kernel
    sumReduction<<<numBlocks, blockSize>>>(d_input, d_partialSums, N);

    // Launch second kernel (1 block, 32 threads for final reduction)
    finalReduction<<<1, 32>>>(d_partialSums, d_finalSum, numBlocks);

    // Copy result back to host
    cudaMemcpy(h_output, d_finalSum, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Total Sum: %d\n", *h_output); // Should print 1 million

    // Free memory
    cudaFree(d_input);
    cudaFree(d_partialSums);
    cudaFree(d_finalSum);
    free(h_input);
    free(h_output);

    return 0;
}
