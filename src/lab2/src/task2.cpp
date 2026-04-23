#include "cli.hpp"
#include "matrix.hpp"
#include "mpi_utils.hpp"

#include <array>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <type_traits>

using std::vector;

struct TaskHeader {
	int n = 0;
	int k = 0;
	int m = 0;
	int procs = 0;
};

MpiDatatype make_header() {
	return make_struct_datatype(
		{ 1, 1, 1, 1 },
		std::array<MPI_Aint, 4>{
			offsetof(TaskHeader, n),
			offsetof(TaskHeader, k),
			offsetof(TaskHeader, m),
			offsetof(TaskHeader, procs),
		},
		{ mpi_type_v<int>, mpi_type_v<int>, mpi_type_v<int>, mpi_type_v<int> });
}

int main(int argc, char** argv) {
	const CliOptions options = parse_cli(argc, argv);

	MPIEnvironment mpi(argc, argv);
	MPIWorld world;

	TaskHeader header{ options.dim, options.dim, options.dim, world.size() };
	const auto header_type = make_header();
	world.bcast_value(header, header_type.get());

	if (header.n % header.procs != 0) {
		throw std::runtime_error("Matrix dimension must be divisible by the process count.");
	}

	const int local_rows = header.n / header.procs;
	vector<double> A, B, C;
	B.resize(header.k * header.m);
	vector<double> local_A(local_rows * header.k);
	vector<double> local_C(local_rows * header.m);

	if (world.rank() == 0) {
		load_input(options.input_dir, header.n, A, B);
		C.resize(header.n * header.m);
	}

	DistributedTimer timer(world);
	const double elapsed = timer.measure([&] {
		world.scatter_equal(A, local_A);

		world.bcast(B);
		const auto A_view = make_matrix_view(local_A, local_rows, header.k);
		const auto B_view = make_matrix_view(B, header.k, header.m);
		auto C_view = make_matrix_view(local_C, local_rows, header.m);
		gemm(A_view, B_view, C_view);

		world.gather_equal(local_C, C);
	});

	if (world.rank() == 0) {
		std::cout << std::fixed << std::setprecision(6) << elapsed << '\n';
		std::filesystem::create_directories(options.output_dir);
		write_matrix(options.output_dir / std::format("C_{}.txt", header.n), header.n, header.m, C);
	}

	return 0;
}
