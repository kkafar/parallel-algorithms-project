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
    int path_count;
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


    const int par_paths_min = 1;
    const int par_paths_max = 4096;
    std::vector<int> paths(par_paths_max - par_paths_min + 1);
    for (int i = par_paths_min; i <= par_paths_max; ++i) {
        paths.push_back(i);
    }

    std::vector<float> dt_list{};
    for (int i = 1; i <= 20; ++i) {
        dt_list.push_back(i * 0.001);
    }

    // Allocate memory on the host
    auto *h_output = new float[par_paths_max];

    // Allocate memory on the device
    float *d_output;
    cudaMalloc((void**)&d_output, par_paths_max * sizeof(float));

    std::vector<sim_result> results{};

    // Run kernel few times before measuring times to eliminate initial outliers
    const int warmup_runs = 48;

    for (int i = 0; i < warmup_runs; ++i) {
        unsigned long long seed = dist(mt);
        langevin_equation<<<par_paths_max, 1>>>(d_output, dt_list[0], gamma, seed);
        cudaMemcpy(h_output, d_output, par_paths_max * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Actual measurements
    for (int path_count = par_paths_min; path_count <= par_paths_max; ++path_count) {
        const int numBlocks = path_count;

        for (float dt : dt_list) {
            unsigned long long seed = dist(mt);
            auto start = std::chrono::high_resolution_clock::now();
            langevin_equation<<<numBlocks, 1>>>(d_output, dt, gamma, seed);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

            // WE DO NOT MEASURE COPY TIME
            // Copy the results back to the host
            cudaMemcpy(h_output, d_output, path_count * sizeof(float), cudaMemcpyDeviceToHost);

            const float total_avg = std::accumulate(h_output, h_output + path_count, 0.0f) / static_cast<float>(path_count);
            const auto square_fn = [total_avg](auto val) { return (val - total_avg) * (val - total_avg); };

            // Notice that I'm dividing by N, not by (N - 1)
            const float stddev = std::sqrt(std::transform_reduce(h_output, h_output + path_count, 0.0f, std::plus{}, square_fn) / (path_count));
            results.push_back({path_count, dt, total_avg, stddev, duration.count()});
        }
    }

    std::printf("path_count,dt,avg,std,time\n");
    for (const auto& result : results) {
        std::printf("%d,%.5f,%.2f,%.2f,%ld\n", result.path_count, result.dt, result.avg_time, result.std_dev, result.duration_us);
    }

    // Free device and host memory
    cudaFree(d_output);
    free(h_output);

    return 0;
}

