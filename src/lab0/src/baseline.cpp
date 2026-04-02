#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
vector<vector<float>> read_matrix(const string& filename) {
	int n = 0, m = 0;
	ifstream fin(filename);
	fin >> n >> m;
	vector<vector<float>> matrix(n, vector<float>(m));
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			fin >> matrix[i][j];
		}
	}
	fin.close();
	return matrix;
}

vector<vector<float>> matrix_multiply(const vector<vector<float>>& A, const vector<vector<float>>& B) {
	vector<vector<float>> C(A.size(), vector<float>(B[0].size()));
	for (int i = 0; i < (int)A.size(); ++i) {
		for (int j = 0; j < (int)B[0].size(); ++j) {
			for (int k = 0; k < (int)A[0].size(); ++k) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	return C;
}

int main() {
	auto A = read_matrix("data/A.txt");
	auto B = read_matrix("data/B.txt");
	auto start = chrono::steady_clock::now();
	auto C = matrix_multiply(A, B);
	auto end = chrono::steady_clock::now();
	ofstream fout("data/C.txt");
	for (int i = 0; i < (int)C.size(); ++i) {
		for (int j = 0; j < (int)C[0].size(); ++j) {
			fout << C[i][j] << ' ';
		}
		fout << '\n';
	}
	fout.close();
	auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	cout << "Time taken: " << duration << " ms" << endl;
	return 0;
}