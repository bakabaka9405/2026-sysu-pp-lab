#include "cli.hpp"
#include "matrix.hpp"
#include "mpi_utils.hpp"

#include <iomanip>
#include <iostream>

using std::vector;

int main(int argc, char** argv) {
	const CliOptions options = parse_cli(argc, argv);

	MPIEnvironment mpi(argc, argv);
	MPIWorld world;

	const int dim = options.dim;
	const int procs = world.size();
	if (dim % procs != 0) {
		throw std::runtime_error("Matrix dimension must be divisible by the process count.");
	}

	const int local_rows = dim / procs;
	vector<double> A;
	vector<double> B;
	vector<double> C;
	B.resize(dim * dim);
	vector<double> local_A(local_rows * dim);
	vector<double> local_C(local_rows * dim);

	if (world.rank() == 0) {
		load_input(options.input_dir, dim, A, B);
		C.resize(dim * dim);
	}

	DistributedTimer timer(world);
	const double elapsed = timer.measure([&] {
		world.scatter_equal(A, local_A);

		world.bcast(B);

		const auto A_view = make_matrix_view(local_A, local_rows, dim);
		const auto B_view = make_matrix_view(B, dim, dim);
		auto C_view = make_matrix_view(local_C, local_rows, dim);
		gemm(A_view, B_view, C_view);

		world.gather_equal(local_C, C);
	});

	if (world.rank() == 0) {
		std::cout << std::fixed << std::setprecision(6) << elapsed << '\n';
		std::filesystem::create_directories(options.output_dir);
		write_matrix(options.output_dir / std::format("C_{}.txt", dim), dim, dim, C);
	}

	return 0;
}
