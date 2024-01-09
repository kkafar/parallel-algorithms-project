#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <ranges>
#include <cmath>
#include <memory>
#include <random>
#include <tuple>

__global__ void langevin_equation(float *output, float dt, float gamma, unsigned long long seed) {
    int idx = blockIdx.x;
    float x = 0.0;
    float v = 0.0;
    float t = 0.0;

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

struct sim_result {
    float dt;
    float avg_time;
    float std_dev;
};


int main() {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<unsigned long long> dist(1, 10000); // for seed generation...

    const float gamma = 0.33;
    int par_paths = 24;
    std::vector<float> dt_list{0.001, 0.002, 0.003};

    // Allocate memory on the host
    float *h_output = new float[par_paths];

    // Allocate memory on the device
    float *d_output;
    cudaMalloc((void**)&d_output, par_paths * sizeof(float));

    int blockSize = 1;
    int numBlocks = (par_paths + blockSize - 1) / blockSize;

    std::vector<sim_result> results{};

    for (float dt : dt_list) {
        unsigned long long seed = dist(mt);
        langevin_equation<<<numBlocks, blockSize>>>(d_output, dt, gamma, seed);

        // Copy the results back to the host
        cudaMemcpy(h_output, d_output, par_paths * sizeof(float), cudaMemcpyDeviceToHost);

        const float total_avg = std::accumulate(h_output, h_output + par_paths, 0.0f) / static_cast<float>(par_paths);
        const auto square_fn = [total_avg](auto val) { return (val - total_avg) * (val - total_avg); };
        const float stddev = std::sqrt(std::transform_reduce(h_output, h_output + par_paths, 0.0f, std::plus{}, square_fn) / (par_paths - 1));
        results.push_back({dt, total_avg, stddev});
    }

    std::printf("dt,avg,std\n");
    for (const auto& result : results) {
        std::printf("%.5f,%.2f,%.2f\n", result.dt, result.avg_time, result.std_dev);
    }
    std::printf("\n");


    // Free device and host memory
    cudaFree(d_output);

    return 0;
}

