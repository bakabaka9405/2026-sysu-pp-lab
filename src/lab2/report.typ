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

#table(
  columns: (0.6fr, 1.9fr, 1fr, 2fr),
  align: center + horizon,
  stroke: 0.5pt,
  inset: 8pt,
  [实验], [基于MPI的并行矩阵乘法（进阶）], [专业（方向）], [计算机科学与技术（人工智能与大数据）],
  [学号], [23336223], [姓名], [万耀文],
  [Email], [#text(size: 10pt, "wanyw3@mail2.sysu.edu.cn")], [完成日期], [2026年4月9日],
)

#v(1em)

#let section-title(num, title) = {
  text(size: 16pt, weight: "bold")[#num. #title]
}

= 实验目的

改进上次实验中的MPI并行矩阵乘法 (MPI-v1)，并讨论不同通信方式对性能的影响。

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

统一的输入数据已提前通过代码随机生成，这里省略不提。

三个入口共用同一套基于 `std::mdspan` 的矩阵视图、切分、乘法和结果回收代码。`task1` 采用均分行块并配合集合通信，`task2` 先把尺寸信息封成结构体，再通过 `mpi_type_create_struct` 对应的包装发送，`task3` 则改进划分方式，以二维划分组织并行计算。

== 核心代码

共用封装层保留了上次实验中的 `MPIEnvironment` 和 `MPIWorld` 结构，并补充了结构体通信、矩阵视图构造以及二维划分所需的通信包装。任务二使用的结构体类型，直接通过 `MPI_Type_create_struct` 生成并提交，然后再用统一接口广播出去。

```cpp
template <std::size_t N>
inline MpiDatatype make_struct_datatype(
	const std::array<int, N>& block_lengths,
	const std::array<MPI_Aint, N>& displacements,
	const std::array<MPI_Datatype, N>& types);

template <typename T>
void bcast_value(T& value, MPI_Datatype type, int root = 0) const;

template <typename T>
void scatterv(const std::vector<T>& send,
			 const std::vector<int>& send_counts,
			 const std::vector<int>& displacements,
			 std::vector<T>& recv,
			 int root = 0) const;
```

矩阵乘法只保留基于 `std::mdspan` 的按块实现。局部矩阵拿到以后，先构造视图，再按行、列和共享维度做三重循环，结果写回局部结果数组。

更新后的实现把矩阵存储与计算接口统一为 `std::mdspan` 视图，并用 concept 限定读写能力。这样在任务一和任务二中可以直接复用同一份乘法内核，在任务三中也可以直接借助 `std::submdspan` 取出二维子块。

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

`task3` 先用 `MPI_Dims_create` 将进程数分解为 `prow × pcol` 的二维网格，再把 A 按进程行切成若干行块，把 B 按进程列切成若干列块。根进程先把对应子块发送到各自的行根或列根，再通过行内广播和列内广播分发到整个进程网格，这样每个进程只负责一个 `C` 子块的计算。

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

三组数据放在一起看，整体趋势比较一致。矩阵越大，并行后节省的时间越明显；矩阵较小时，进程启动和通信的开销占比更高，所以不同做法之间差别不大。第二种实现把控制参数封成结构体后，整体时间与第一种仍然接近，说明参数封装带来的额外代价很小。第三种实现采用二维划分后，在较大规模下更容易获得稳定收益，尤其是在 1024 和 2048 规模、进程数较高时，结果明显优于前两种实现，这说明二维分块能够更好地平衡行与列方向的通信和计算压力。

= 实验感想

这次实验把三种实现放在同一套共用代码下，代码结构更清楚，也更容易比较。最后看到的结果说明，矩阵规模足够大时，分配到多个进程后确实能把总时间压下来，但当规模较小时，通信和调度的影响还是很明显。整体来看，先把共用部分整理好，再去改通信方式，实验过程会更稳一些。
