#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

struct CliOptions final {
	int dim = 2048;
	std::filesystem::path input_dir = "data";
	std::filesystem::path output_dir = "data";
};

 CliOptions parse_cli(int argc, char** argv) {
	CliOptions options;

	for (int i = 1; i < argc; ++i) {
		const std::string_view arg = argv[i];

		if (arg == "--dim") {
			if (i + 1 >= argc) {
				throw std::runtime_error("Missing value after --dim.");
			}
			options.dim = std::stoi(argv[++i]);
			continue;
		}

		if (arg == "--input-dir") {
			if (i + 1 >= argc) {
				throw std::runtime_error("Missing value after --input-dir.");
			}
			options.input_dir = argv[++i];
			continue;
		}

		if (arg == "--output-dir") {
			if (i + 1 >= argc) {
				throw std::runtime_error("Missing value after --output-dir.");
			}
			options.output_dir = argv[++i];
			continue;
		}

		if (!arg.empty() && arg.front() != '-') {
			options.dim = std::stoi(argv[i]);
			continue;
		}

		throw std::runtime_error("Unknown argument: " + std::string(arg));
	}

	return options;
}
