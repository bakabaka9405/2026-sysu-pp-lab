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
	int n = (int)A.size(), m = (int)A[0].size(), p = (int)B[0].size();
	vector<vector<float>> C(n, vector<float>(p));
	for (int i = 0; i < n; ++i) {
		for (int k = 0; k < m; ++k) {
			float tmp = A[i][k];
			for (int j = 0; j < p % 4; ++j) {
				C[i][j] += tmp * B[k][j];
			}
			for (int j = p % 4; j < p; j += 4) {
				C[i][j] += tmp * B[k][j];
				C[i][j + 1] += tmp * B[k][j + 1];
				C[i][j + 2] += tmp * B[k][j + 2];
				C[i][j + 3] += tmp * B[k][j + 3];
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
	ofstream fout("data/output.txt");
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