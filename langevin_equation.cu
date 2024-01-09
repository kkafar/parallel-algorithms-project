#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <ranges>
#include <cmath>

__global__ void langevin_equation(float *output, int n, float gamma, unsigned long long seed) {
    int idx = blockIdx.x;
    float x = 0.0;
    float v = 0.0;
    float t = 0.0;
    float dt = 0.001;

    curandState state;
    curand_init(seed, idx, 0, &state);

    while (true) {
        v = v + (-gamma * v * dt + curand_normal(&state));
        x = x + v * dt;
        t += dt;

        if ((x >= 0 ? x : -x ) > 1) {
            break;
        }
    }

    output[idx] = t;
}

int main() {
    float gamma = 0.33;

    // Set the size of the array
    int n = 24;

    // Allocate memory on the host
    float *h_output = (float*)malloc(n * sizeof(float));

    // Allocate memory on the device
    float *d_output;
    cudaMalloc((void**)&d_output, n * sizeof(float));

    // Set mean, standard deviation, and seed
    unsigned long long seed = 1234;

    // Launch the CUDA kernel
    int blockSize = 1;
    int numBlocks = (n + blockSize - 1) / blockSize;

    langevin_equation<<<numBlocks, blockSize>>>(d_output, n, gamma, seed);

    // Copy the results back to the host
    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    const float total_avg = std::accumulate(h_output, h_output + n, 0.0f) / static_cast<float>(n);
    const auto square_fn = [total_avg](auto val) { return (val - total_avg) * (val - total_avg); };
    const float stddev = std::sqrt(std::transform_reduce(h_output, h_output + n, 0.0f, std::plus{}, square_fn) / (n - 1));

    for (int i = 0; i < n; ++i) {
        printf("%f ", h_output[i]);
    }
    printf("\n");
    printf("Avg: %.2f with std: %.2f\n", total_avg, stddev);

    // Free device and host memory
    cudaFree(d_output);
    free(h_output);

    return 0;
}

