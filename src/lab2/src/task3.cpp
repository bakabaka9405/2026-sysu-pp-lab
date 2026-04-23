#include "cli.hpp"
#include "matrix.hpp"
#include "mpi_utils.hpp"

#include <iomanip>
#include <iostream>
#include <utility>

using std::vector;

namespace {

vector<double> pack_view(MatrixReadable auto const& view) {
	vector<double> packed(view.extent(0) * view.extent(1));
	std::size_t p = 0;

	for (std::size_t i = 0; i < view.extent(0); ++i) {
		for (std::size_t j = 0; j < view.extent(1); ++j) {
			packed[p++] = view[i, j];
		}
	}

	return packed;
}

void unpack_view(MatrixWritable auto view, const vector<double>& packed) {
	std::size_t p = 0;

	for (std::size_t i = 0; i < view.extent(0); ++i) {
		for (std::size_t j = 0; j < view.extent(1); ++j) {
			view[i, j] = packed[p++];
		}
	}
}

auto slice_block(MatrixReadable auto const& view,
				 std::size_t row_begin,
				 std::size_t row_count,
				 std::size_t col_begin,
				 std::size_t col_count) {
	return std::submdspan(view,
						  std::pair{ row_begin, row_begin + row_count },
						  std::pair{ col_begin, col_begin + col_count });
}

} // namespace

int main(int argc, char** argv) {
	const CliOptions options = parse_cli(argc, argv);

	MPIEnvironment mpi(argc, argv);
	MPIWorld world;

	const int dim = options.dim;
	const int procs = world.size();

	std::array<int, 2> dims{ 0, 0 };
	mpi_check(MPI_Dims_create(procs, 2, dims.data()), "MPI_Dims_create");
	const int prow = dims[0];
	const int pcol = dims[1];

	const int row = world.rank() / pcol;
	const int col = world.rank() % pcol;

	MPI_Comm row_comm = MPI_COMM_NULL;
	MPI_Comm col_comm = MPI_COMM_NULL;
	mpi_check(MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm), "MPI_Comm_split");
	mpi_check(MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm), "MPI_Comm_split");

	MPIWorld row_world(row_comm);
	MPIWorld col_world(col_comm);

	const vector<int> row_size = split_rows(dim, prow);
	const vector<int> row_begin = prefix_sum(row_size);
	const vector<int> col_size = split_rows(dim, pcol);
	const vector<int> col_begin = prefix_sum(col_size);

	vector<double> A;
	vector<double> B;
	vector<double> C;
	vector<double> local_A(row_size[row] * dim);
	vector<double> local_B(dim * col_size[col]);
	vector<double> local_C(row_size[row] * col_size[col]);

	if (world.rank() == 0) {
		load_input(options.input_dir, dim, A, B);
		C.resize(dim * dim);
	}

	DistributedTimer timer(world);
	const double elapsed = timer.measure([&] {
		constexpr int a_tag = 1;
		constexpr int b_tag = 2;
		constexpr int c_tag = 3;

		if (world.rank() == 0) {
			const auto A_view = make_matrix_view(A, dim, dim);
			const auto B_view = make_matrix_view(B, dim, dim);

			for (int r = 0; r < prow; ++r) {
				auto A_block = slice_block(A_view, row_begin[r], row_size[r], 0, dim);
				auto packed = pack_view(A_block);

				if (r == 0) local_A = std::move(packed);
				else world.send(packed, r * pcol, a_tag);
			}

			for (int c = 0; c < pcol; ++c) {
				auto B_block = slice_block(B_view, 0, dim, col_begin[c], col_size[c]);
				auto packed = pack_view(B_block);

				if (c == 0) local_B = std::move(packed);
				else world.send(packed, c, b_tag);
			}
		}
		else {
			if (col == 0) world.recv(local_A, 0, a_tag);
			if (row == 0) world.recv(local_B, 0, b_tag);
		}

		row_world.bcast(local_A);
		col_world.bcast(local_B);

		const auto A_view = make_matrix_view(local_A, row_size[row], dim);
		const auto B_view = make_matrix_view(local_B, dim, col_size[col]);
		auto C_view = make_matrix_view(local_C, row_size[row], col_size[col]);
		gemm(A_view, B_view, C_view);

		if (world.rank() == 0) {
			const auto C_view_full = make_matrix_view(C, dim, dim);
			unpack_view(slice_block(C_view_full, 0, row_size[0], 0, col_size[0]), local_C);

			for (int r = 0; r < prow; ++r) {
				for (int c = 0; c < pcol; ++c) {
					if (r == 0 && c == 0) continue;

					vector<double> block(row_size[r] * col_size[c]);
					world.recv(block, r * pcol + c, c_tag);
					unpack_view(slice_block(C_view_full,
											row_begin[r],
											row_size[r],
											col_begin[c],
											col_size[c]),
								block);
				}
			}
		}
		else {
			world.send(local_C, 0, c_tag);
		}
	});

	if (world.rank() == 0) {
		std::cout << std::fixed << std::setprecision(6) << elapsed << '\n';
		std::filesystem::create_directories(options.output_dir);
		write_matrix(options.output_dir / std::format("C_{}.txt", dim), dim, dim, C);
	}

	mpi_check(MPI_Comm_free(&row_comm), "MPI_Comm_free");
	mpi_check(MPI_Comm_free(&col_comm), "MPI_Comm_free");

	return 0;
}
