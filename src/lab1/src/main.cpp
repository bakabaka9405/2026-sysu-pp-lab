#include "mpi_utils.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace fs = filesystem;

void load_matrix(int k, vector<double>& A, vector<double>& B) {
	const fs::path pathA = fs::path("data") / ("A_" + to_string(k) + ".txt");
	const fs::path pathB = fs::path("data") / ("B_" + to_string(k) + ".txt");

	ifstream fileA(pathA);
	ifstream fileB(pathB);

	for (double& v : A) fileA >> v;
	for (double& v : B) fileB >> v;
}

void write_matrix(int rows, int cols, const vector<double>& C, const fs::path& path) {
	ofstream out(path);
	if (!out.is_open()) {
		throw runtime_error("Failed to open output matrix file.");
	}

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			out << C[i * cols + j] << ' ';
		}
		out << '\n';
	}
}

void multiply_local(int m,
					int k,
					int n,
					const vector<double>& A,
					const vector<double>& B,
					vector<double>& C) {
	ranges::fill(C, 0.0);
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			for (int t = 0; t < k; ++t) {
				C[i * n + j] += A[i * k + t] * B[t * n + j];
			}
		}
	}
}

int main(int argc, char* argv[]) {
	MPIEnvironment mpi(argc, argv);
	const MPIWorld world;

	const int rank = world.rank();
	const int size = world.size();

	const int m = 2048;
	const int k = m;
	const int n = m;

	if (rank == 0) {
		cout << "size:" << size << endl;
	}

	const int local_rows = m / size;

	vector<double> A;
	vector<double> B(k * n);
	vector<double> C;
	vector<double> local_A(local_rows * k);
	vector<double> local_C(local_rows * n, 0.0);

	if (rank == 0) {
		A.resize(m * k);
		C.resize(m * n);
	}

	DistributedTimer timer(world);

	if (rank == 0) {
		load_matrix(k, A, B);
	}

	timer.measure("main", [&]() {
		const int tag_scatter = 100;
		const int tag_bcast = 200;
		const int tag_gather = 300;

		if (rank == 0) {
			copy_n(A.begin(), local_A.size(), local_A.begin());
			for (int peer = 1; peer < size; ++peer) {
				vector<double> block(local_A.size());
				copy_n(A.begin() + (std::ptrdiff_t)(block.size() * peer),
					   block.size(), block.begin());
				world.send(block, peer, tag_scatter);
			}
		}
		else {
			world.recv(local_A, 0, tag_scatter);
		}

		if (rank == 0) {
			for (int peer = 1; peer < size; ++peer) {
				world.send(B, peer, tag_bcast);
			}
		}
		else {
			world.recv(B, 0, tag_bcast);
		}

		multiply_local(local_rows, k, n, local_A, B, local_C);

		if (rank == 0) {
			copy_n(local_C.begin(), local_C.size(), C.begin());
			for (int peer = 1; peer < size; ++peer) {
				vector<double> block(local_C.size());
				world.recv(block, peer, tag_gather);
				copy_n(block.begin(), block.size(),
					   C.begin() + std::ptrdiff_t(peer * block.size()));
			}
		}
		else {
			world.send(local_C, 0, tag_gather);
		}
	});

	if (rank == 0) {
		cout << timer.records()[0].seconds << endl;
		write_matrix(m, n, C, fs::path("data") / ("C_" + to_string(k) + ".txt"));
	}
	return 0;
}