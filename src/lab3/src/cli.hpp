#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace fs = std::filesystem;

enum class Workload {
	Matrix,
	Array,
};

struct CliOptions final {
	int dim = 128;
	std::size_t size = 1ULL << 20;
	std::string size_label = "1M";
	int threads = 1;
	fs::path input_dir;
	fs::path output_dir;
};

std::size_t parse_size(std::string_view value) {
	if (value.empty()) {
		throw std::runtime_error("Array size must not be empty.");
	}

	std::size_t multiplier = 1;
	if (value.back() == 'M' || value.back() == 'm') {
		multiplier = 1ULL << 20;
		value.remove_suffix(1);
	}

	std::size_t number = 0;
	const auto* first = value.data();
	const auto* last = first + value.size();
	const auto [ptr, ec] = std::from_chars(first, last, number);
	if (ec != std::errc{} || ptr != last || number == 0) {
		throw std::runtime_error("Invalid array size.");
	}

	return number * multiplier;
}

std::string size_label(std::size_t size) {
	if (size % (1ULL << 20) == 0) {
		return std::to_string(size / (1ULL << 20)) + "M";
	}
	return std::to_string(size);
}

CliOptions parse_cli(int argc, char** argv, Workload workload) {
	CliOptions options;
	if (workload == Workload::Matrix) {
		options.input_dir = "data/matrix";
		options.output_dir = "data/matrix";
	}
	else {
		options.input_dir = "data/array";
		options.output_dir = "data/array";
	}

	for (int i = 1; i < argc; ++i) {
		const std::string_view arg = argv[i];
		if (arg == "--") {
			continue;
		}

		if (arg == "--dim") {
			if (i + 1 >= argc) throw std::runtime_error("Missing value after --dim.");
			options.dim = std::stoi(argv[++i]);
			continue;
		}

		if (arg == "--size") {
			if (i + 1 >= argc) throw std::runtime_error("Missing value after --size.");
			options.size = parse_size(argv[++i]);
			options.size_label = size_label(options.size);
			continue;
		}

		if (arg == "--threads") {
			if (i + 1 >= argc) throw std::runtime_error("Missing value after --threads.");
			options.threads = std::stoi(argv[++i]);
			continue;
		}

		if (arg == "--input-dir") {
			if (i + 1 >= argc) throw std::runtime_error("Missing value after --input-dir.");
			options.input_dir = argv[++i];
			continue;
		}

		if (arg == "--output-dir") {
			if (i + 1 >= argc) throw std::runtime_error("Missing value after --output-dir.");
			options.output_dir = argv[++i];
			continue;
		}

		if (!arg.empty() && arg.front() != '-') {
			if (workload == Workload::Matrix) {
				options.dim = std::stoi(argv[i]);
			}
			else {
				options.size = parse_size(arg);
				options.size_label = size_label(options.size);
			}
			continue;
		}

		throw std::runtime_error("Unknown argument: " + std::string(arg));
	}

	if (options.threads <= 0) {
		throw std::runtime_error("Thread count must be positive.");
	}

	return options;
}
