#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <random>
#include <vector>

struct ThreadTask final {
	int tid = 0;
	std::size_t count = 0;
	unsigned seed = 0;
	std::size_t inside_count = 0;
	std::exception_ptr error;
};

void* worker(void* arg) {
	auto& task = *(ThreadTask*)(arg);
	try {
		std::mt19937 rng(task.seed);
		std::uniform_real_distribution<double> dist(-1.0, 1.0);

		std::size_t count = 0;
		for (std::size_t i = 0; i < task.count; ++i) {
			const double x = dist(rng);
			const double y = dist(rng);
			if (x * x + y * y <= 1.0) {
				++count;
			}
		}
		task.inside_count = count;
	}
	catch (...) {
		task.error = std::current_exception();
	}
	return nullptr;
}

void pthread_check(int code, std::string_view operation) {
	if (code != 0) {
		throw std::runtime_error(std::string(operation) + " failed.");
	}
}

int main(int argc, char** argv) {
	if (argc != 3) return 1;

	const int n = std::stoi(argv[1], nullptr, 10);
	const int thread_count = std::stoi(argv[2]);

	if (n == 0 || thread_count <= 0) return 1;

	std::vector<pthread_t> threads(thread_count);
	std::vector<ThreadTask> tasks(thread_count);

	const unsigned base_seed = std::random_device{}();

	const auto start = std::chrono::steady_clock::now();

	for (int tid = 0; tid < thread_count; ++tid) {
		tasks[tid].tid = tid;
		tasks[tid].count = n / thread_count
						   + (tid < n % thread_count ? 1 : 0);
		tasks[tid].seed = base_seed + static_cast<unsigned>(tid);
		pthread_check(
			pthread_create(&threads[tid], nullptr, worker, &tasks[tid]),
			"pthread_create");
	}

	std::size_t total_inside = 0;
	for (int tid = 0; tid < thread_count; ++tid) {
		pthread_check(pthread_join(threads[tid], nullptr), "pthread_join");
		if (tasks[tid].error) std::rethrow_exception(tasks[tid].error);
		total_inside += tasks[tid].inside_count;
	}

	const auto end = std::chrono::steady_clock::now();
	const double elapsed = std::chrono::duration<double>(end - start).count();

	const double pi_estimate = 4.0 * (double)total_inside / (double)n;
	std::cout << std::fixed << std::setprecision(6);
	std::cout << n << ' ' << total_inside << ' ' << pi_estimate << ' ' << elapsed << std::endl;

	return 0;
}
