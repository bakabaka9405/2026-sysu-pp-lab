#pragma once

#include <concepts>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <mdspan>
#include <stdexcept>
#include <vector>

using MatrixExtents = std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent>;
using MatrixView = std::mdspan<double, MatrixExtents, std::layout_right>;
using ConstMatrixView = std::mdspan<const double, MatrixExtents, std::layout_right>;

std::filesystem::path matrix_file(const std::filesystem::path& dir, char prefix, int dim) {
	return dir / std::format("{}_{}.txt", prefix, dim);
}

void load_matrix(const std::filesystem::path& path, std::vector<double>& data) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open matrix file: " + path.string());
	}

	for (auto& value : data) {
		file >> value;
	}
}

void load_input(const std::filesystem::path& dir, int dim, std::vector<double>& A, std::vector<double>& B) {
	A.resize(dim * dim);
	B.resize(dim * dim);

	load_matrix(matrix_file(dir, 'A', dim), A);
	load_matrix(matrix_file(dir, 'B', dim), B);
}

MatrixView make_matrix_view(std::vector<double>& data, int rows, int cols) {
	return MatrixView(data.data(), rows, cols);
}

ConstMatrixView make_matrix_view(const std::vector<double>& data, int rows, int cols) {
	return ConstMatrixView(data.data(), rows, cols);
}

void write_matrix(const std::filesystem::path& path, int rows, int cols, const std::vector<double>& C) {
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
	const auto n = A.extent(0);
	const auto k = A.extent(1);
	const auto m = B.extent(1);

	if (k != B.extent(0) || n != C.extent(0) || m != C.extent(1)) {
		throw std::runtime_error("Matrix shapes do not match for multiplication.");
	}

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			C[i, j] = 0.0;
		}

		for (int t = 0; t < k; ++t) {
			const double tmp = A[i, t];
			for (int j = 0; j < m; ++j) {
				C[i, j] += tmp * B[t, j];
			}
		}
	}
}
