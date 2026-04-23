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
	int rows = 0;
	int shared = 0;
	int cols = 0;
	int procs = 0;
};

MpiDatatype make_task_header_type() {
	return make_struct_datatype(std::array<int, 4>{ 1, 1, 1, 1 },
								std::array<MPI_Aint, 4>{
									static_cast<MPI_Aint>(offsetof(TaskHeader, rows)),
									static_cast<MPI_Aint>(offsetof(TaskHeader, shared)),
									static_cast<MPI_Aint>(offsetof(TaskHeader, cols)),
									static_cast<MPI_Aint>(offsetof(TaskHeader, procs)),
								},
								std::array<MPI_Datatype, 4>{ mpi_type_v<int>, mpi_type_v<int>, mpi_type_v<int>, mpi_type_v<int> });
}

int main(int argc, char** argv) {
	const CliOptions options = parse_cli(argc, argv);

	MPIEnvironment mpi(argc, argv);
	MPIWorld world;

	TaskHeader header{ options.dim, options.dim, options.dim, world.size() };
	const auto header_type = make_task_header_type();
	world.bcast_value(header, header_type.get());

	if (header.rows % header.procs != 0) {
		throw std::runtime_error("Matrix dimension must be divisible by the process count.");
	}

	const int local_rows = header.rows / header.procs;
	vector<double> A;
	vector<double> B;
	vector<double> C;
	B.resize(header.shared * header.cols);
	vector<double> local_A(local_rows * header.shared);
	vector<double> local_C;

	if (world.rank() == 0) {
		load_input(options.input_dir, header.rows, A, B);
		C.resize(header.rows * header.cols);
	}

	DistributedTimer timer(world);
	timer.measure("task2", [&] {
		world.scatter_equal(A, local_A);

		world.bcast(B);
		multiply_block(local_rows, header.shared, header.cols, local_A, B, local_C);

		world.gather_equal(local_C, C);
	});

	if (world.rank() == 0) {
		std::cout << std::fixed << std::setprecision(6) << timer.records().front().seconds << '\n';
		std::filesystem::create_directories(options.output_dir);
		write_matrix(options.output_dir / std::format("C_{}.txt", header.rows), header.rows, header.cols, C);
	}

	return 0;
}
