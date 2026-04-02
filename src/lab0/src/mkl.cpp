#include <chrono>
#include <fstream>
#include <iostream>
#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

using namespace std;

int main() {
	auto async_handler = [](const sycl::exception_list& exceptions) {
		for (auto const& e : exceptions) {
			try {
				std::rethrow_exception(e);
			}
			catch (sycl::exception const& ex) {
				cerr << "Asynchronous SYCL exception:\n"
					 << ex.what() << endl;
			}
		}
	};
	sycl::queue q{ sycl::default_selector_v, async_handler };
	int m = 0, k = 0, k_ = 0, n = 0;

	ifstream in("data/A.txt");
	in >> m >> k;

	auto* A = sycl::malloc_shared<double>(m * k, q);
	for (int i = 0; i < m * k; i++)
		in >> A[i];
	in.close();

	in.open("data/B.txt");
	in >> k_ >> n;
	assert(k == k_);

	auto* B = sycl::malloc_shared<double>(k * n, q);
	for (int i = 0; i < k * n; i++)
		in >> B[i];
	in.close();

	auto* C = sycl::malloc_shared<double>(m * n, q);
	for (int i = 0; i < m * n; i++)
		C[i] = 0.0;

	auto start = chrono::high_resolution_clock::now();

	oneapi::mkl::blas::row_major::gemm(
		q,
		oneapi::mkl::transpose::nontrans,
		oneapi::mkl::transpose::nontrans,
		m, n, k,
		1.0,
		A, k,
		B, n,
		0.0,
		C, n);

	q.wait();

	auto end = chrono::high_resolution_clock::now();

	ofstream out("data/output.txt");
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++)
			out << C[i * n + j] << " ";
		out << endl;
	}
	out.close();

	sycl::free(A, q);
	sycl::free(B, q);
	sycl::free(C, q);

	cout << "Device: "
		 << q.get_device().get_info<sycl::info::device::name>() << endl;
	cout << "Time taken: "
		 << chrono::duration_cast<chrono::milliseconds>(end - start).count()
		 << " ms" << endl;

	return 0;
}