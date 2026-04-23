#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;
namespace fs = filesystem;

void gen(const fs::path& dst, int n, int m, default_random_engine& e) {
	uniform_real_distribution<double> u(0, 1);
	ofstream fout(dst);
	for (int i = 0; i < n * m; ++i) {
		fout << u(e) << ' ';
		if ((i + 1) % m == 0) {
			fout << '\n';
		}
	}
	fout.close();
}

int main() {
	default_random_engine e(
		chrono::steady_clock::now().time_since_epoch().count());
	uniform_real_distribution<double> u(0, 1);

	fs::create_directory("data");
	gen("data/A_128.txt", 128, 128, e);
	gen("data/A_256.txt", 256, 256, e);
	gen("data/A_512.txt", 512, 512, e);
	gen("data/A_1024.txt", 1024, 1024, e);
	gen("data/A_2048.txt", 2048, 2048, e);
	gen("data/B_128.txt", 128, 128, e);
	gen("data/B_256.txt", 256, 256, e);
	gen("data/B_512.txt", 512, 512, e);
	gen("data/B_1024.txt", 1024, 1024, e);
	gen("data/B_2048.txt", 2048, 2048, e);
	return 0;
}