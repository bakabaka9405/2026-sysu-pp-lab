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
  [实验], [基于MPI的并行矩阵乘法（进阶）], [专业（方向）], [计算机科学与技术（人工智能与大数据）],
[学号], [#id], [姓名], [#author],
[Email], [#text(size: 10pt, email)], [完成日期], [2026年4月24日],
)

#v(1em)

#let section-title(num, title) = {
  text(size: 16pt, weight: "bold")[#num. #title]
}

= 实验目的

改进 MPI 并行矩阵乘法 (MPI-v1)，并讨论不同通信方式对性能的影响。

*输入*：$m,n,k$ 三个整数，每个整数的取值范围均为 $[128, 2048]$

*问题描述*：随机生成 $m times n$ 的矩阵 $A$ 及 $n times k$的矩阵 $B$，并对这两个矩阵进行矩阵乘法运算，得到矩阵 $C$.

*输出*：$A,B,C$ 三个矩阵，及矩阵计算所消耗的时间 $t$。

*要求*：
1. 三种实现方式：
  1. 采用MPI集合通信实现并行矩阵乘法中的进程间通信；
  2. 使用 `mpi_type_create_struct` 聚合 MPI 进程内变量后通信（例如矩阵尺寸 $m, n, k$ 或者其他变量）；
  3. 尝试不同数据/任务划分方式（选做）。
2. 对于不同实现方式，调整并记录不同线程数量（1-16）及矩阵规模（128-2048）下的时间开销，填写下页表格，并分析其性能及扩展性。

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
- 编译器版本：gcc version 16.0.1 20260222 (experimental) (MinGW-W64 x86_64-ucrt-posix-seh, built by Brecht Sanders, r1)

== 实验设计

为了确保编译器优化不会将矩阵乘法过程省略，所有benchmark均以从文件读取两个矩阵、向文件输出结果矩阵的形式运行，保证每次运行的输入相同，计时仅包含计算过程，不包含输入输出过程。

统一的输入数据已提前通过代码随机生成，并在不同实现之间保持一致。

三个入口共用同一套矩阵视图构造、数据切分和乘法内核。`task1` 采用均分行块配合集合通信，`task2` 先将尺寸信息封为结构体再通过 `MPI_Type_create_struct` 对应的派生类型广播，`task3` 则采用二维进程网格组织并行计算。

== 核心代码

矩阵乘法内核基于 `std::mdspan` 视图实现，以 concept 限定读写能力，三个任务复用同一份乘法逻辑。循环顺序为 `i`→`t`→`j`，将 `A[i,t]` 提至最内层循环外以减少重复访问。

```cpp
template <class View>
concept MatrixReadable = requires(const View& view, std::size_t i, std::size_t j) {
	{ view.extent(0) } -> std::convertible_to<std::size_t>;
	{ view.extent(1) } -> std::convertible_to<std::size_t>;
	{ view[i, j] } -> std::convertible_to<double>;
};

template <class View>
concept MatrixWritable = MatrixReadable<View> && requires(View view, std::size_t i, std::size_t j, double value) {
	view[i, j] = value;
};

void gemm(MatrixReadable auto const& A, MatrixReadable auto const& B, MatrixWritable auto C) {
	const std::size_t n = A.extent(0);
	const std::size_t k = A.extent(1);
	const std::size_t m = B.extent(1);

	if (k != B.extent(0) || n != C.extent(0) || m != C.extent(1)) {
		throw std::runtime_error("Matrix shapes do not match for multiplication.");
	}

	for (std::size_t i = 0; i < n; i++)
		for (std::size_t j = 0; j < m; j++)
			C[i, j] = 0.0;

	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t t = 0; t < k; ++t) {
			const double tmp = A[i, t];
			for (std::size_t j = 0; j < m; ++j) {
				C[i, j] += tmp * B[t, j];
			}
		}
	}
}
```

`task2` 使用的 `TaskHeader` 结构体将矩阵维度和进程数打包为一个整体，通过 `MPI_Type_create_struct` 构造派生类型后广播。

```cpp
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
```

=== 任务一

设进程数为 $p$，矩阵边长为 $d$，每个进程处理 $d/p$ 行，局部结果大小为 $(d/p) times d$。

根进程读取输入后，通过 `MPI_Scatter` 将矩阵 $A$ 按行均分到各进程，再将完整的 $B$ 广播给所有进程。各进程在局部行块上调用乘法内核，最后用 `MPI_Gather` 回收结果。

```cpp
const int local_rows = dim / procs;
vector<double> local_A(local_rows * dim);
vector<double> local_C(local_rows * dim);

world.scatter_equal(A, local_A);
world.bcast(B);

const auto A_view = make_matrix_view(local_A, local_rows, dim);
const auto B_view = make_matrix_view(B, dim, dim);
auto C_view = make_matrix_view(local_C, local_rows, dim);
gemm(A_view, B_view, C_view);

world.gather_equal(local_C, C);
```

=== 任务二

任务二的通信模式与任务一的区别在于控制参数的传播方式：矩阵维度 $n,k,m$ 及进程数封装进上文已定义的 `TaskHeader`，通过 `MPI_Type_create_struct` 生成的派生类型一次性广播到所有进程。

```cpp
TaskHeader header{ options.dim, options.dim, options.dim, world.size() };
const auto header_type = make_header();
world.bcast_value(header, header_type.get());

const int local_rows = header.n / header.procs;
...
world.scatter_equal(A, local_A);
world.bcast(B);
...
gemm(A_view, B_view, C_view);
world.gather_equal(local_C, C);
```

=== 任务三

任务三将进程组织为二维网格。设总进程数为 $P$，通过 `MPI_Dims_create` 分解为 $P = p_r times p_c$，其中 `p_r` 为进程行数，`p_c` 为进程列数。矩阵 $A$（$m times n$）按行切分为 $p_r$ 块，矩阵 $B$（$n times k$）按列切分为 $p_c$ 块，进程 $(r,c)$ 负责计算子块 $C_{r,c} = A_r B_c$。

设第 $r$ 个行块大小为 $m_r times n$，第 $c$ 个列块大小为 $n times k_c$，局部计算量为 $O(m_r n k_c)$。均衡划分下 $m_r ≈ m / p_r, k_c ≈ k / p_c$，每个进程计算量近似 $O((m n k) / P)$。

```cpp
std::array<int, 2> dims{ 0, 0 };
mpi_check(MPI_Dims_create(procs, 2, dims.data()), "MPI_Dims_create");
const int prow = dims[0];
const int pcol = dims[1];

const int row = world.rank() / pcol;
const int col = world.rank() % pcol;

MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);
```

`MPI_Comm_split` 按行编号和列编号分别创建行通信域和列通信域，同一行的进程通过行内广播共享 $A$ 行块，同一列的进程通过列内广播共享 $B$ 列块。

由于 `std::submdspan` 取出的子块不是连续存储，发送前用 `pack_view` 整理为连续缓冲区，接收后用 `unpack_view` 写回对应子块位置。

```cpp
const auto row_counts = split_rows(dim, prow);
const auto col_counts = split_rows(dim, pcol);
const auto row_displs = prefix_sum(row_counts);
const auto col_displs = prefix_sum(col_counts);

const int row = world.rank() / pcol;
const int col = world.rank() % pcol;

MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);

auto A_block = slice_block(A_view, row_displs[r], row_counts[r], 0, dim);
auto B_block = slice_block(B_view, 0, dim, col_displs[c], col_counts[c]);
```

每个进程拿到局部子块后调用上文定义的 `gemm` 得到 $C_{r,c}$。根进程通过点对点通信逐块回收所有子块，按偏移写回全局 $C$ 矩阵，非根进程直接将局部结果发回根进程。

= 实验结果

第一种实现的结果如下。

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
  [1], [0.000362], [0.002539], [0.018575], [0.162845], [2.248916],
  [2], [0.000311], [0.001723], [0.011035], [0.097654], [1.286171],
  [4], [0.000593], [0.001204], [0.006543], [0.102224], [0.961546],
  [8], [0.000733], [0.001752], [0.005966], [0.105539], [0.836732],
  [16], [0.001196], [0.001831], [0.007437], [0.107725], [0.844656],
)

第一种实现为作为 baseline。128 矩阵规模下，单进程耗时 0.000362s，16 进程反而升至 0.001196s，通信与同步开销在问题规模过小时占主导。512 规模开始出现稳定加速：单进程 0.018575s 对 8 进程 0.005966s。2048 规模下加速效果最为显著，从单进程 2.248916s 降至 8 进程 0.836732s，16 进程 0.844656s 与 8 进程基本持平，说明 I/O 和同步瓶颈开始显现。

第二种实现的结果如下。

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
  [1], [0.000360], [0.002511], [0.018850], [0.161080], [2.180046],
  [2], [0.000478], [0.001753], [0.010169], [0.103269], [1.306407],
  [4], [0.000568], [0.001130], [0.006103], [0.095631], [0.955449],
  [8], [0.000455], [0.001873], [0.007136], [0.103104], [0.969275],
  [16], [0.000723], [0.001468], [0.007613], [0.112654], [0.902465],
)

第二种实现各规模、各进程数下与第一种相差均在毫秒量级以内——2048 规模 16 进程下第二种 0.902465s 对第一种 0.844656s，差值属正常测量抖动。说明将控制参数封装为派生类型广播不引入可观测的额外开销。

第三种实现的结果如下。

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
  [1], [0.000439], [0.002736], [0.020197], [0.168723], [2.234676],
  [2], [0.000334], [0.001594], [0.010922], [0.104662], [1.330520],
  [4], [0.000499], [0.001638], [0.008288], [0.059984], [1.045352],
  [8], [0.000825], [0.001077], [0.008299], [0.062985], [0.870710],
  [16], [0.001792], [0.001727], [0.008498], [0.044372], [0.812574],
)

第三种实现的表现在不同规模下分化明显。128 和 256 规模下，pack/unpack 及额外的通信组织使耗时略高于前两种实现。1024 规模开始出现优势：4 进程 0.059984s 对第二种的 0.095631s，16 进程 0.044372s 对第二种的 0.112654s。2048 规模 16 进程下取得 0.812574s，为三种实现中最低。二维划分将通信压力分散到行广播和列广播两条独立路径，在大规模、高进程数场景下通信开销更低。

整体来看，矩阵规模越小，进程启动和通信开销占比越高，不同实现之间差异不大。规模增大后，第一种和第二种在 1024、2048 规模下相比单进程有数倍加速，第三种在同样规模、高进程数下进一步优于前两者。二维网格划分在处理大规模矩阵时能更有效地平衡行与列方向的通信负载。

= 实验感想

二维划分写起来比一维麻烦不少，但性能提升不少。c++26 很好用，但是GCC 16 正式版什么时候发。