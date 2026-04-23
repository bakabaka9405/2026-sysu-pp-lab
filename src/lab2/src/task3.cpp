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
	const vector<int> row_counts = split_rows(dim, procs);
	const vector<int> row_displs = prefix_sum(row_counts);
	const vector<int> a_counts = scale_counts(row_counts, dim);
	const vector<int> a_displs = scale_counts(row_displs, dim);
	const vector<int> c_counts = scale_counts(row_counts, dim);
	const vector<int> c_displs = scale_counts(row_displs, dim);

	vector<double> A;
	vector<double> B;
	vector<double> C;
	B.resize(dim * dim);
	vector<double> local_A(a_counts[world.rank()]);
	vector<double> local_C(c_counts[world.rank()]);

	if (world.rank() == 0) {
		load_input(options.input_dir, dim, A, B);
		C.resize(dim * dim);
	}

	DistributedTimer timer(world);
	timer.measure("task3", [&] {
		world.scatterv(A, a_counts, a_displs, local_A);
		world.bcast(B);

		const int local_rows = row_counts[world.rank()];
		multiply_block(local_rows, dim, dim, local_A, B, local_C);

		world.gatherv(local_C, c_counts, c_displs, C);
	});

	if (world.rank() == 0) {
		std::cout << std::fixed << std::setprecision(6) << timer.records().front().seconds << '\n';
		std::filesystem::create_directories(options.output_dir);
		write_matrix(options.output_dir / std::format("C_{}.txt", dim), dim, dim, C);
	}

	return 0;
}
