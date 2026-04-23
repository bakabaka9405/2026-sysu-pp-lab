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
	const MatrixShape shape{ dim, dim, dim };
	vector<double> A;
	vector<double> B;
	vector<double> C;
	B.resize(dim * dim);
	vector<double> local_A(local_rows * dim);
	vector<double> local_C;

	if (world.rank() == 0) {
		load_input(options.input_dir, dim, A, B);
		C.resize(dim * dim);
	}

	DistributedTimer timer(world);
	timer.measure("task1", [&] {
		world.scatter_equal(A, local_A);

		world.bcast(B);

		multiply_block(local_rows, shape.shared, shape.cols, local_A, B, local_C);

		world.gather_equal(local_C, C);
	});

	if (world.rank() == 0) {
		std::cout << std::fixed << std::setprecision(6) << timer.records().front().seconds << '\n';
		std::filesystem::create_directories(options.output_dir);
		write_matrix(options.output_dir / std::format("C_{}.txt", dim), dim, dim, C);
	}

	return 0;
}
