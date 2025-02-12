#include <stdio.h>

__global__ void sumReduction(int *input, int *output, int N) {
    __shared__ int sharedMem[256]; // Shared memory for block-level reduction

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    // Load elements into shared memory
    sharedMem[tid] = (index < N) ? input[index] : 0;
    __syncthreads(); // Ensure all threads have loaded their values before proceeding

    // Perform parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        __syncthreads(); // Synchronize after each reduction step
    }

    // Store the block's partial sum in global memory
    if (tid == 0) {
        output[blockIdx.x] = sharedMem[0];
    }
}

int main() {
    int N = 1024; // Array size
    int blockSize = 256; // Threads per block
    int numBlocks = (N + blockSize - 1) / blockSize; // Grid size

    int *h_input, *h_output;
    int *d_input, *d_output;

    // Allocate memory on host
    h_input = (int*) malloc(N * sizeof(int));
    h_output = (int*) malloc(numBlocks * sizeof(int));

    // Initialize input array
    for (int i = 0; i < N; i++) {
        h_input[i] = 1; // Example: all elements are 1, so sum should be N
    }

    // Allocate memory on GPU
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, numBlocks * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    sumReduction<<<numBlocks, blockSize>>>(d_input, d_output, N);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Final summation on CPU
    int totalSum = 0;
    for (int i = 0; i < numBlocks; i++) {
        totalSum += h_output[i];
    }

    printf("Total Sum: %d\n", totalSum); // Should print 1024 if all elements are 1

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
