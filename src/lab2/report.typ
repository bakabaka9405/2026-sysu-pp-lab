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

== 实验设计

三个入口共用同一套矩阵读入、切分、乘法和结果回收代码。`task1` 采用均分行块并配合集合通信，`task2` 先把尺寸信息封成结构体，再通过 `mpi_type_create_struct` 对应的包装发送，`task3` 则使用不等分发的行块划分，让不同进程拿到的行数不完全相同。

== 核心代码

共用封装层保留了上次实验中的 `MPIEnvironment` 和 `MPIWorld` 结构，并补充了结构体通信与变长分发的包装。任务二使用的结构体类型，直接通过 `MPI_Type_create_struct` 生成并提交，然后再用统一接口广播出去。

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

矩阵乘法只保留按块计算的一份实现。局部矩阵拿到以后，直接按行、列和共享维度做三重循环，结果写回局部结果数组。

```cpp
inline void multiply_block(int rows, int shared, int cols,
						   const vector<double>& A,
						   const vector<double>& B,
						   vector<double>& C) {
	C.assign(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), 0.0);
	for (int i = 0; i < rows; ++i) {
		for (int t = 0; t < shared; ++t) {
			for (int j = 0; j < cols; ++j) {
				C[static_cast<std::size_t>(i) * cols + j] +=
					A[static_cast<std::size_t>(i) * shared + t] *
					B[static_cast<std::size_t>(t) * cols + j];
			}
		}
	}
}
```

`task3` 先按进程数算出每个进程拿到多少行，再把行数换成元素数交给 `scatterv` 和 `gatherv`。这样即使某些规模不能整除进程数，也能直接处理。

```cpp
const auto row_counts = split_rows(dim, procs);
const auto row_displs = prefix_sum(row_counts);
const auto a_counts = scale_counts(row_counts, dim);
const auto a_displs = scale_counts(row_displs, dim);
world.scatterv(A, a_counts, a_displs, local_A);
world.gatherv(local_C, c_counts, c_displs, C);
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
  [1], [0.000438], [0.002642], [0.019492], [0.156743], [2.139832],
  [2], [0.000338], [0.001641], [0.009977], [0.096904], [1.316903],
  [4], [0.000611], [0.001469], [0.006078], [0.088599], [0.905714],
  [8], [0.000520], [0.001784], [0.007062], [0.081866], [0.844810],
  [16], [0.000976], [0.002219], [0.007385], [0.101666], [0.814141],
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
  [1], [0.000381], [0.002633], [0.019162], [0.158397], [2.124900],
  [2], [0.000521], [0.001908], [0.015639], [0.090618], [1.253576],
  [4], [0.000556], [0.001599], [0.006700], [0.088080], [2.111655],
  [8], [0.000710], [0.001689], [0.005952], [0.080984], [0.799958],
  [16], [0.001037], [0.001917], [0.006970], [0.103926], [0.806941],
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
  [1], [0.001609], [0.002470], [0.018704], [0.155879], [2.229869],
  [2], [0.000454], [0.001425], [0.010104], [0.095341], [3.640236],
  [4], [0.000729], [0.001819], [0.006985], [0.090474], [2.110239],
  [8], [0.000924], [0.001548], [0.005834], [0.082977], [0.795026],
  [16], [0.001696], [0.003165], [0.010533], [0.099046], [0.840918],
)

三组数据放在一起看，整体趋势比较一致。矩阵越大，并行后节省的时间越明显；矩阵较小时，进程启动和通信的开销占比更高，所以不同做法之间差别不大。第二种实现把控制参数封成结构体后，整体时间和第一种很接近，说明这部分开销不算大。第三种实现换成不等分发后，整体也没有明显变慢，只是在个别点上有一些波动，说明这次测试主要瓶颈还是矩阵计算本身。

= 实验感想

这次实验把三种实现放在同一套共用代码下，代码结构更清楚，也更容易比较。最后看到的结果说明，矩阵规模足够大时，分配到多个进程后确实能把总时间压下来，但当规模较小时，通信和调度的影响还是很明显。整体来看，先把共用部分整理好，再去改通信方式，实验过程会更稳一些。
