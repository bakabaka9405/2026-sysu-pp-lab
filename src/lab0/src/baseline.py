from pathlib import Path
import time


def read_matrix(path: Path) -> list[list[float]]:
    with open(path) as f:
        next(f)  # Skip the first line
        return [list(map(float, line.split())) for line in f]


def write_matrix(path: Path, matrix: list[list[float]]):
    with open(path, "w") as f:
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")


def matrix_multiply(A, B):
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C


def main():
    A = read_matrix(Path("data/A.txt"))
    B = read_matrix(Path("data/B.txt"))
    start = time.perf_counter()
    C = matrix_multiply(A, B)
    end = time.perf_counter()
    write_matrix(Path("data/output.txt"), C)
    print(f"Time taken: {(end - start) * 1000:.2f} ms")


if __name__ == "__main__":
    main()
