#include "cli.hpp"
#include "matrix.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct ThreadTask final {
	const std::vector<double>* A = nullptr;
	const std::vector<double>* B = nullptr;
	std::vector<double>* C = nullptr;
	int dim = 0;
	std::size_t row_begin = 0;
	std::size_t row_end = 0;
	std::exception_ptr error;
};

void* worker(void* arg) {
	auto& task = *static_cast<ThreadTask*>(arg);
	try {
		const auto A_view = make_matrix_view(*task.A, task.dim, task.dim);
		const auto B_view = make_matrix_view(*task.B, task.dim, task.dim);
		auto C_view = make_matrix_view(*task.C, task.dim, task.dim);
		const auto A_block = std::submdspan(A_view, std::pair{ task.row_begin, task.row_end }, std::full_extent);
		auto C_block = std::submdspan(C_view, std::pair{ task.row_begin, task.row_end }, std::full_extent);
		gemm(A_block, B_view, C_block);
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

} // namespace

int main(int argc, char** argv) {
	const CliOptions options = parse_cli(argc, argv, Workload::Matrix);

	std::vector<double> A;
	std::vector<double> B;
	std::vector<double> C(static_cast<std::size_t>(options.dim) * options.dim);
	load_input(options.input_dir, options.dim, A, B);

	std::vector<pthread_t> threads(static_cast<std::size_t>(options.threads));
	std::vector<ThreadTask> tasks(static_cast<std::size_t>(options.threads));

	const auto begin = std::chrono::steady_clock::now();
	for (int tid = 0; tid < options.threads; ++tid) {
		const std::size_t row_begin = static_cast<std::size_t>(options.dim) * tid / options.threads;
		const std::size_t row_end = static_cast<std::size_t>(options.dim) * (tid + 1) / options.threads;
		tasks[tid] = ThreadTask{ &A, &B, &C, options.dim, row_begin, row_end, {} };
		pthread_check(pthread_create(&threads[tid], nullptr, worker, &tasks[tid]), "pthread_create");
	}

	for (int tid = 0; tid < options.threads; ++tid) {
		pthread_check(pthread_join(threads[tid], nullptr), "pthread_join");
		if (tasks[tid].error) {
			std::rethrow_exception(tasks[tid].error);
		}
	}
	const auto end = std::chrono::steady_clock::now();

	std::filesystem::create_directories(options.output_dir);
	write_matrix(options.output_dir / std::format("C_{}.txt", options.dim), options.dim, options.dim, C);

	const double elapsed = std::chrono::duration<double>(end - begin).count();
	std::cout << std::fixed << std::setprecision(6) << elapsed << '\n';
	return 0;
}
