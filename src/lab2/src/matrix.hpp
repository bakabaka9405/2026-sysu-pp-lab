#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using std::string;
using std::vector;
namespace fs = std::filesystem;

struct MatrixShape final {
	int rows = 0;
	int shared = 0;
	int cols = 0;
};

fs::path matrix_file(const fs::path& dir, char prefix, int dim) {
	return dir / std::format("{}_{}.txt", prefix, dim);
}

void load_matrix(const fs::path& path, vector<double>& data) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open matrix file: " + path.string());
	}

	for (auto& i : data) file >> i;
}

void load_input(const fs::path& dir, int dim, vector<double>& A, vector<double>& B) {
	A.resize(dim * dim);
	B.resize(dim * dim);

	const fs::path path_a = matrix_file(dir, 'A', dim);
	const fs::path path_b = matrix_file(dir, 'B', dim);

	load_matrix(path_a, A);
	load_matrix(path_b, B);
}

void write_matrix(const fs::path& path, int rows, int cols, const vector<double>& C) {
	std::ofstream out(path);
	if (!out.is_open()) {
		throw std::runtime_error("Failed to open matrix output file: " + path.string());
	}

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			out << C[i * cols + j] << ' ';
		}
		out << '\n';
	}
}

vector<int> split_rows(int rows, int procs) {
	vector<int> cnt(procs, rows / procs);
	for (int i = 0; i < rows % procs; ++i) {
		++cnt[i];
	}
	return cnt;
}

vector<int> prefix_sum(const vector<int>& cnt) {
	vector<int> displacements(cnt.size(), 0);
	for (std::size_t i = 1; i < cnt.size(); ++i) {
		displacements[i] = displacements[i - 1] + cnt[i - 1];
	}
	return displacements;
}

vector<int> scale_counts(const vector<int>& cnt, int factor) {
	vector<int> scaled(cnt.size(), 0);
	for (std::size_t i = 0; i < cnt.size(); ++i) {
		scaled[i] = cnt[i] * factor;
	}
	return scaled;
}

void multiply_block(int rows,
					int shared,
					int cols,
					const vector<double>& A,
					const vector<double>& B,
					vector<double>& C) {
	C.assign(rows * cols, 0.0);

	for (int i = 0; i < rows; ++i) {
		double* c_row = C.data() + i * cols;
		const double* a_row = A.data() + i * shared;

		for (int t = 0; t < shared; ++t) {
			const double a_value = a_row[t];
			const double* b_row = B.data() + t * cols;

			for (int j = 0; j < cols; ++j) {
				c_row[j] += a_value * b_row[j];
			}
		}
	}
}
