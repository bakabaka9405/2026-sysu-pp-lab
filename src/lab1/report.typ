#set page(margin: (x: 2.5cm, y: 2cm))
#set text(font: "Noto Serif CJK SC", size: 12pt)
#set par(leading: 1em)
#set heading(numbering: "1.")

#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()
#codly(languages: codly-languages)

#align(center)[
  #text(font: "Noto Serif CJK SC", size: 18pt, weight: "bold")[中山大学计算机学院本科生实验报告]
  #v(0.3em)
  #text(size: 14pt, weight: "bold")[（2025 学年春季学期）]
]

#v(0.5em)

#grid(
  columns: (1fr, 1fr),
  [课程名称：并行程序设计与算法], align(right)[批改人：#h(12em)],
)

#v(0.3em)

#let author = sys.inputs.at("author", default: "数据删除")
#let id = sys.inputs.at("id", default: "数据删除")
#let email = sys.inputs.at("email", default: "数据删除")

#table(
  columns: (0.6fr, 1.9fr, 1fr, 2fr),
  align: center + horizon,
  stroke: 0.5pt,
  inset: 8pt,
  [实验], [基于MPI的并行矩阵乘法], [专业（方向）], [计算机科学与技术（人工智能与大数据）],
[学号], [#id], [姓名], [#author],
[Email], [#text(size: 10pt, email)], [完成日期], [2026年4月3日],
)

#v(1em)

#let section-title(num, title) = {
  text(size: 16pt, weight: "bold")[#num. #title]
}

= 实验目的

使用 MPI 点对点通信方式实现并行通用矩阵乘法 (MPI-v1)，并通过实验分析不同进程数量、矩阵规模时该实现的性能。

*输入*：$m,n,k$ 三个整数，每个整数的取值范围均为 $[128, 2048]$

*问题描述*：随机生成 $m times n$ 的矩阵 $A$ 及 $n times k$的矩阵 $B$，并对这两个矩阵进行矩阵乘法运算，得到矩阵 $C$.

*输出*：$A,B,C$ 三个矩阵，及矩阵计算所消耗的时间 $t$。

*要求*：
1. 使用MPI点对点通信实现并行矩阵乘法，调整并记录不同线程数量（1-16）及矩阵规模（128-2048）下的时间开销，填写下页表格，并分析其性能。
2. 根据当前实现，在实验报告中讨论两个优化方向：a) 在内存有限的情况下，如何进行大规模矩阵乘法计算？b) 如何提高大规模稀疏矩阵乘法性能？


#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  align: center + horizon,
  inset: 8pt,
  table.cell(
    rowspan: 2,
    align: center,
  )[
    进程数
  ],
  table.cell(
    colspan: 5,
    align: center + horizon,
  )[
    矩阵规模
  ],
  [128], [256], [512], [1024], [2048],
  [1], [], [], [], [], [],
  [2], [], [], [], [], [],
  [4], [], [], [], [], [],
  [8], [], [], [], [], [],
  [16], [], [], [], [], [],
)

= 实验过程和核心代码

== 实验平台配置

- 操作系统：Windows 11 23H2 22631.6199
- 处理器：13th Gen Intel(R) Core(TM) i5-13600K 12 核 20 线程
- Microsoft MPI 版本：10.1.12498.18

== 实验设计
为了确保编译器优化不会将矩阵乘法过程省略，所有benchmark均以从文件读取两个矩阵、向文件输出结果矩阵的形式运行，计时仅包含计算过程，不包含输入输出过程。

本次实验使用的数据规模：两个 $128 times 128$ 到 $1024 times 1024$ 的浮点数矩阵，值域为 $[0,1]$。

统一的输入数据已提前通过代码随机生成，这里省略不提。

== 核心代码

我设计了一套封装函数，使用 RAII 技术将 MPI 上下文限制在函数作用域内，防止中途报出异常时没有及时调用 `MPI_Finalize()`：

```cpp
class MPIEnvironment final {
public:
	MPIEnvironment(int& argc, char**& argv) {
		mpi_check(MPI_Init(&argc, &argv), "MPI_Init");
	}

	~MPIEnvironment() noexcept {
		int finalized = 0;
		MPI_Finalized(&finalized);
		if (!finalized) {
			MPI_Finalize();
		}
	}

	MPIEnvironment(const MPIEnvironment&) = delete;
	MPIEnvironment& operator=(const MPIEnvironment&) = delete;
	MPIEnvironment(MPIEnvironment&&) = delete;
	MPIEnvironment& operator=(MPIEnvironment&&) = delete;
};
```

在该封装之上继续抽象通信上下文，统一管理进程编号、进程总数与常用点对点操作：

```cpp
class MPIWorld final {
public:
  explicit MPIWorld(MPI_Comm comm = MPI_COMM_WORLD)
    : comm_(comm) {
    mpi_check(MPI_Comm_rank(comm_, &rank_), "MPI_Comm_rank");
    mpi_check(MPI_Comm_size(comm_, &size_), "MPI_Comm_size");
  }

  int rank() const noexcept { return rank_; }
  int size() const noexcept { return size_; }

  template <typename T>
  void send(const std::vector<T>& data, int dest, int tag = 0) const {
    mpi_check(
      MPI_Send(data.data(), int(data.size()), mpi_type_v<T>,
           dest, tag, comm_),
      "MPI_Send");
  }

  template <typename T>
  void recv(std::vector<T>& data, int src, int tag = 0) const {
    mpi_check(
      MPI_Recv(data.data(), int(data.size()), mpi_type_v<T>,
           src, tag, comm_, MPI_STATUS_IGNORE),
      "MPI_Recv");
  }

private:
  MPI_Comm comm_;
  int rank_ = 0;
  int size_ = 1;
};
```

计时部分采用“全局同步 + 根进程记录最大耗时”的策略。每个阶段开始前后都执行一次屏障，从而保证各进程在统一边界上开始和结束；随后用 `MPI_Reduce` 归约得到该阶段真实耗时，这个值等价于并行程序端到端耗时。

```cpp
template <typename Func>
void measure(const std::string& stage, Func&& task) {
  world_.barrier();
  WallTimer timer;
  std::forward<Func>(task)();
  world_.barrier();
  const double stage_seconds = world_.reduce_max(timer.elapsed_seconds());
  if (world_.rank() == 0) {
    records_.emplace_back(stage, stage_seconds);
  }
}
```

矩阵乘法本体采用按行切分。设总进程数为 $p$，矩阵 $A$ 行数为 $m$，则每个进程处理 $"local_rows" = m/p$ 行。根进程先保留自己的首块数据，再把后续块按进程序号发送给对应进程。这样可以把计算任务映射为多个独立子问题，每个子问题都执行同一份局部乘法逻辑。

```cpp
const int local_rows = m / size;
vector<double> local_A(local_rows * k);

if (rank == 0) {
  copy_n(A.begin(), local_A.size(), local_A.begin());
  for (int peer = 1; peer < size; ++peer) {
    vector<double> block(local_A.size());
    copy_n(A.begin() + (std::ptrdiff_t)(block.size() * peer),
         block.size(), block.begin());
    world.send(block, peer, tag_scatter);
  }
} else {
  world.recv(local_A, 0, tag_scatter);
}
```

由于每个子问题都需要完整矩阵 $B$，实现中由根进程向其他进程逐一发送 $B$，其通信语义与广播一致，但实现手段保持为点对点通信，从而满足实验要求。

```cpp
if (rank == 0) {
  for (int peer = 1; peer < size; ++peer) {
    world.send(B, peer, tag_bcast);
  }
} else {
  world.recv(B, 0, tag_bcast);
}
```

每个进程拿到局部行块后执行三重循环完成局部矩阵乘法，局部结果形状为 $"local_rows" times n$。时间复杂度为 $O("local_rows" dot k dot n)$。

```cpp
void multiply_local(int m, int k, int n,
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
```

最后由根进程按进程号回收局部结果并写回全局矩阵 $C$ 的对应区间，非根进程只负责发送自己的局部块：

```cpp
if (rank == 0) {
  copy_n(local_C.begin(), local_C.size(), C.begin());
  for (int peer = 1; peer < size; ++peer) {
    vector<double> block(local_C.size());
    world.recv(block, peer, tag_gather);
    copy_n(block.begin(), block.size(),
         C.begin() + std::ptrdiff_t(peer * block.size()));
  }
} else {
  world.send(local_C, 0, tag_gather);
}
```


= 实验结果

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  align: center + horizon,
  inset: 8pt,
  table.cell(
    rowspan: 2,
    align: center,
  )[
    进程数
  ],
  table.cell(
    colspan: 5,
    align: center + horizon,
  )[
    矩阵规模
  ],
  [128], [256], [512], [1024], [2048],
  [1], [0.0009714], [0.0077833], [0.100913], [2.17428], [58.3383],
  [2], [0.0006664], [0.0043267], [0.0514589], [1.17683], [29.5109],
  [4], [0.0005792], [0.002539], [0.0314815], [0.732136], [15.6614],
  [8], [0.0011559], [0.0035969], [0.0229782], [0.421768], [8.51037],
  [16], [0.0019594], [0.0044639], [0.021541], [0.277365], [6.57435],
)

结果表明，随着矩阵规模增大，并行计算的优势逐渐显现，运行时间整体明显下降，说明 MPI 能较好提升大规模矩阵乘法的计算效率。进程数从 1 增加到 8 或 16 时，中大规模矩阵的加速效果较为突出；但在小规模情况下，性能提升有限，甚至可能略有波动。这说明通信与调度开销在小问题中占比较高，而在大问题中并行性更容易得到发挥。

= 实验感想



最后谈一下实验要求中提到的两个优化问题：

== 在内存有限的情况下，如何进行大规模矩阵乘法计算？
矩阵规模大到内存无法容纳时，考虑进行分块矩阵乘法。将矩阵 $A$ 和 $B$ 分成更小的子块，逐块加载到内存中进行乘法计算，并将结果写回磁盘。这样可以在内存有限的情况下处理任意规模的矩阵，但会引入更多的磁盘 I/O 开销，性能可能会受到较大影响。

== 如何提高大规模稀疏矩阵乘法性能？
对于大规模稀疏矩阵乘法，考虑使用除了方阵外的更适合稀疏矩阵的存储格式，如只保存非 0 元素的值与坐标，乘法过程中也只计算非 0 元素的乘积，从而大幅减少计算量和内存占用。