#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <mdspan>
#include <stdexcept>
#include <string>
#include <vector>

using std::string;
using std::vector;
namespace fs = std::filesystem;

using MatrixExtents = std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent>;
using MatrixView = std::mdspan<double, MatrixExtents, std::layout_right>;
using ConstMatrixView = std::mdspan<const double, MatrixExtents, std::layout_right>;

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

MatrixView make_matrix_view(vector<double>& data, int rows, int cols) {
	return MatrixView(data.data(), rows, cols);
}

ConstMatrixView make_matrix_view(const vector<double>& data, int rows, int cols) {
	return ConstMatrixView(data.data(), rows, cols);
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

template <class View>
concept MatrixReadable = requires(const View& view, std::size_t i, std::size_t j) {
	{ view.extent(0) } -> std::convertible_to<std::size_t>;
	{ view.extent(1) } -> std::convertible_to<std::size_t>;
	{ view[i, j] } -> std::convertible_to<double>;
};

template <class View>
concept MatrixWritable = MatrixReadable<View> && requires(View view, std::size_t i, std::size_t j, double value) {
	view[i, j] = value;
};

void gemm(MatrixReadable auto const& A, MatrixReadable auto const& B, MatrixWritable auto C) {
	const std::size_t n = A.extent(0);
	const std::size_t k = A.extent(1);
	const std::size_t m = B.extent(1);

	if (k != B.extent(0) || n != C.extent(0) || m != C.extent(1)) {
		throw std::runtime_error("Matrix shapes do not match for multiplication.");
	}

	for (std::size_t i = 0; i < n; i++)
		for (std::size_t j = 0; j < m; j++)
			C[i, j] = 0.0;

	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t t = 0; t < k; ++t) {
			const double tmp = A[i, t];
			for (std::size_t j = 0; j < m; ++j) {
				C[i, j] += tmp * B[t, j];
			}
		}
	}
}
