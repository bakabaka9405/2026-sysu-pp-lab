#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;
namespace fs = filesystem;

int main() {
	int n = 0, m = 0, k = 0;
	cin >> n >> m >> k;
	default_random_engine e(
		chrono::steady_clock::now().time_since_epoch().count());
	uniform_real_distribution<double> u(0, 1);

	fs::create_directory("data");
	ofstream fout("data/A.txt");
	fout << n << ' ' << m << '\n';
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			fout << u(e) << ' ';
		}
		fout << '\n';
	}
	fout.close();

	fout.open("data/B.txt");
	fout << m << ' ' << k << '\n';
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
			fout << u(e) << ' ';
		}
		fout << '\n';
	}
	fout.close();
	return 0;
}