#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256  // Threads per block

// First kernel: Block-wise reduction
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

// Second kernel: Final sum of partial sums.
__global__ void finalReduction(int *input, int *output, int N) {
    __shared__ int sharedMem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = tid;

    // Load partial sums into shared memory.
    sharedMem[tid] = (index < N) ? input[index] : 0;
    __syncthreads();

    // Perform reduction.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        __syncthreads();
    }

    // Store final sum in global memory.
    if (tid == 0) {
        *output = sharedMem[0];
    }
}

int main() {
    int N = 1 << 20;  // 1 Million elements
    int blockSize = BLOCK_SIZE;
    int numBlocks = (N + blockSize - 1) / blockSize;

    int *h_input, *h_output;
    int *d_input, *d_partialSums, *d_finalSum;

    // Allocate host memory.
    h_input = (int*) malloc(N * sizeof(int));
    h_output = (int*) malloc(sizeof(int));

    // Initialize input array.
    for (int i = 0; i < N; i++) {
        h_input[i] = 1;  // All elements are 1.
    }

    // Allocate GPU memory.
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_partialSums, numBlocks * sizeof(int));
    cudaMalloc(&d_finalSum, sizeof(int));

    // Copy data to GPU.
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Create events for timing.
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

    // Launch second kernel (using 1 block) and time it.
    cudaEventRecord(start);
    finalReduction<<<1, blockSize>>>(d_partialSums, d_finalSum, numBlocks);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime2, start, stop);
    printf("Time elapsed for finalReduction kernel: %f ms\n", elapsedTime2);

    // Copy result back to host.
    cudaMemcpy(h_output, d_finalSum, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Total Sum: %d\n", *h_output); // Should print 1 million

    // Free memory.
    cudaFree(d_input);
    cudaFree(d_partialSums);
    cudaFree(d_finalSum);
    free(h_input);
    free(h_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
