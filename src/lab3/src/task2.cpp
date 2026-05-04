#include "big_integer.hpp"
#include "cli.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t flush_interval = 1ULL << 20;
constexpr std::size_t buffer_size = 1ULL << 20;
constexpr std::size_t record_size = sizeof(std::int32_t);
static_assert(record_size == 4);

struct ThreadTask final {
	std::filesystem::path path;
	std::size_t begin = 0;
	std::size_t end = 0;
	BigInteger sum;
	std::exception_ptr error;
};

void flush_chunk(BigInteger& sum, std::int64_t& chunk, std::size_t& count) {
	if (count == 0) return;
	sum.add(chunk);
	chunk = 0;
	count = 0;
}

void* worker(void* arg) {
	auto& task = *static_cast<ThreadTask*>(arg);
	try {
		std::ifstream in(task.path, std::ios::binary);
		if (!in.is_open()) {
			throw std::runtime_error("Failed to open array input file: " + task.path.string());
		}

		in.seekg(static_cast<std::streamoff>(task.begin * record_size), std::ios::beg);
		if (!in) {
			throw std::runtime_error("Failed to seek array input file: " + task.path.string());
		}

		std::vector<std::int32_t> buffer(buffer_size);
		std::int64_t chunk = 0;
		std::size_t count = 0;
		std::size_t remaining = task.end - task.begin;
		while (remaining > 0) {
			const std::size_t take = std::min(remaining, buffer.size());
			in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(take * record_size));
			if (in.gcount() != static_cast<std::streamsize>(take * record_size)) {
				throw std::runtime_error("Unexpected end of array input file: " + task.path.string());
			}

			for (std::size_t i = 0; i < take; ++i) {
				chunk += buffer[i];
				++count;
				if (count == flush_interval) {
					flush_chunk(task.sum, chunk, count);
				}
			}

			remaining -= take;
		}
		flush_chunk(task.sum, chunk, count);
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
	const CliOptions options = parse_cli(argc, argv, Workload::Array);
	const std::filesystem::path input = options.input_dir / ("A_" + options.size_label + ".in");

	std::vector<pthread_t> threads(options.threads);
	std::vector<ThreadTask> tasks(options.threads);

	const auto begin = std::chrono::steady_clock::now();
	for (int tid = 0; tid < options.threads; ++tid) {
		const std::size_t block_begin = options.size * tid / options.threads;
		const std::size_t block_end = options.size * (tid + 1) / options.threads;
		tasks[tid].path = input;
		tasks[tid].begin = block_begin;
		tasks[tid].end = block_end;
		pthread_check(pthread_create(&threads[tid], nullptr, worker, &tasks[tid]), "pthread_create");
	}

	BigInteger total;
	for (int tid = 0; tid < options.threads; ++tid) {
		pthread_check(pthread_join(threads[tid], nullptr), "pthread_join");
		if (tasks[tid].error) {
			std::rethrow_exception(tasks[tid].error);
		}
		total.add(tasks[tid].sum);
	}
	const auto end = std::chrono::steady_clock::now();

	std::filesystem::create_directories(options.output_dir);
	std::ofstream out(options.output_dir / ("S_" + options.size_label + ".txt"));
	out << total.to_string() << '\n';

	const double elapsed = std::chrono::duration<double>(end - begin).count();
	std::cout << std::fixed << std::setprecision(6) << elapsed << '\n';
	return 0;
}
