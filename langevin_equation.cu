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
#include <algorithm>
#include <chrono>

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
    long duration_us;
};


int main() {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<unsigned long long> dist(1, 10000); // for seed generation...

    const float gamma = 0.33;
    const int par_paths = 24;

    std::vector<float> dt_list{};
    for (int i = 1; i <= 20; ++i) {
        dt_list.push_back(i * 0.001);
    }

    // Allocate memory on the host
    auto *h_output = new float[par_paths];

    // Allocate memory on the device
    float *d_output;
    cudaMalloc((void**)&d_output, par_paths * sizeof(float));

    const int blockSize = 1;
    const int numBlocks = (par_paths + blockSize - 1) / blockSize;

    std::vector<sim_result> results{};

    // Run kernel few times before measuring times to eliminate initial outliers
    const int warmup_runs = 48;

    for (int i = 0; i < warmup_runs; ++i) {
        unsigned long long seed = dist(mt);
        langevin_equation<<<numBlocks, blockSize>>>(d_output, dt_list[0], gamma, seed);
        cudaMemcpy(h_output, d_output, par_paths * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Actual measurements
    for (float dt : dt_list) {
        unsigned long long seed = dist(mt);
        auto start = std::chrono::high_resolution_clock::now();
        langevin_equation<<<numBlocks, blockSize>>>(d_output, dt, gamma, seed);

        // Copy the results back to the host
        cudaMemcpy(h_output, d_output, par_paths * sizeof(float), cudaMemcpyDeviceToHost);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        const float total_avg = std::accumulate(h_output, h_output + par_paths, 0.0f) / static_cast<float>(par_paths);
        const auto square_fn = [total_avg](auto val) { return (val - total_avg) * (val - total_avg); };
        const float stddev = std::sqrt(std::transform_reduce(h_output, h_output + par_paths, 0.0f, std::plus{}, square_fn) / (par_paths - 1));
        results.push_back({dt, total_avg, stddev, duration.count()});
    }

    std::printf("dt,avg,std,time\n");
    for (const auto& result : results) {
        std::printf("%.5f,%.2f,%.2f,%ld\n", result.dt, result.avg_time, result.std_dev, result.duration_us);
    }

    // Free device and host memory
    cudaFree(d_output);
    free(h_output);

    return 0;
}

