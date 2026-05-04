#set page(margin: (x: 2.5cm, y: 2cm))
#set text(font: "Noto Serif CJK SC", size: 12pt)
#set par(leading: 1em)
#set heading(numbering: "1.")

#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()
#codly(languages: codly-languages)

#set page(numbering: "1")

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
  [实验], [Pthreads并行矩阵乘法与数组求和], [专业（方向）], [计算机科学与技术（人工智能与大数据）],
  [学号], [#id], [姓名], [#author],
  [Email], [#text(size: 10pt, email)], [完成日期], [2026年4月24日],
)

#v(1em)

#let section-title(num, title) = {
  text(size: 16pt, weight: "bold")[#num. #title]
}

= 实验目的

== 并行矩阵乘法
使用 `Pthreads` 实现并行矩阵乘法，并通过实验分析其性能。

输入：$m,n,k$ 三个整数，每个整数的取值范围均为 $[128, 2048]$

问题描述：随机生成 $m times n$ 的矩阵 $A$ 及 $n times k$ 的矩阵 $B$，并对这两个矩阵进行矩阵乘法运算，得到矩阵 $C$.

输出：$A,B,C$ 三个矩阵，及矩阵计算所消耗的时间 $t$。

要求：

1. 使用 `Pthread` 创建多线程实现并行矩阵乘法，调整线程数量（1-16）及矩阵规模（128-2048），根据结果分析其并行性能（包括但不限于，时间、效率、可扩展性）。
2. 选做：可分析不同数据及任务划分方式的影响。

== 并行数组求和

使用 `Pthreads` 实现并行数组求和，并通过实验分析其性能。

输入：整数n，取值范围为 [1M, 128M]。

问题描述：随机生成长度为 $n$ 的整型数组 $A$，计算其元素和 $s=sum_(i=1)^n A_i $。

输出：数组A，元素和s，及求和计算所消耗的时间t。

要求：

1. 使用 `Pthreads` 实现并行数组求和，调整线程数量（1-16）及数组规模（1M, 128M），根据结果分析其并行性能（包括但不限于，时间、效率、可扩展性）。
2. 选做：可分析不同聚合方式的影响.



= 实验过程和核心代码

== 实验平台配置

- 操作系统：Windows 11 23H2 22631.6199
- 处理器：13th Gen Intel(R) Core(TM) i5-13600K 12 核 20 线程
- 编译器：gcc version 16.0.1 20260222 (experimental) (MinGW-W64 x86_64-ucrt-posix-seh, built by Brecht Sanders, r1)

== 实验设计

本实验的矩阵乘法部分沿用前一实验中从文件读取固定随机矩阵的方式，通过命令行参数控制矩阵规模和线程数。计时区间只覆盖并行矩阵乘法本身，不包含矩阵输入和结果输出。

数据划分方面，矩阵 $C$ 按行划分为若干连续区间，每个线程只写入自己负责的行段，矩阵 $A$ 与 $B$ 在所有线程之间只读共享，计算过程中无需互斥锁。

数组求和部分同样使用随机生成的固定数组文件作为输入，数组规模和线程数同样由控制台参数控制。为了避免将数组一次性搬入内存，数组文件被设计为二进制定长记录，每个元素按 4 字节 `int32_t` 写入。线程 $i$ 根据自己的元素起止位置直接计算文件偏移并独立打开文件读取，起始偏移为 $4 times "begin"_i$ 字节。由于读取行为本身是该算法的一部分，数组求和的计时区间覆盖线程内文件读取、分块累加与最终归并。

数组求和的结果使用高精度整数保存。线程内部先用 `int64_t` 聚合一段固定长度的数据块，再将块内结果累加到高精度整数中，以降低高精度运算调用频率，同时避免总和超过 `long long` 表示范围时发生溢出。最终主线程在所有工作线程结束后，将各线程的高精度局部和相加并写入结果文件。

== 核心代码

矩阵乘法的核心仍然基于 `std::mdspan` 表示矩阵视图，并通过 concept 约束矩阵视图的读写能力。乘法内核为通用 `gemm` 接口，线程工作函数在调用前用 `std::submdspan` 包裹出 $A$ 与 $C$ 的连续行块。循环顺序采用 $i->t->j$，在进入最内层循环前取出 `A[i,t]`，减少重复寻址。

```cpp
void gemm(MatrixReadable auto const& A, MatrixReadable auto const& B, MatrixWritable auto C) {
	const std::size_t n = A.extent(0);
	const std::size_t k = A.extent(1);
	const std::size_t m = B.extent(1);

	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < m; ++j) {
			C[i, j] = 0.0;
		}

		for (std::size_t t = 0; t < k; ++t) {
			const double tmp = A[i, t];
			for (std::size_t j = 0; j < m; ++j) {
				C[i, j] += tmp * B[t, j];
			}
		}
	}
}
```

`Pthreads` 的任务划分在创建线程前完成。第 `tid` 个线程处理的行范围由整数比例式直接给出，因此矩阵规模不能被线程数整除时仍然能够获得相差不超过一行的负载划分。线程工作函数将完整矩阵视图切分为局部行块，再调用通用乘法内核。

```cpp
const auto A_view = make_matrix_view(*task.A, task.dim, task.dim);
const auto B_view = make_matrix_view(*task.B, task.dim, task.dim);
auto C_view = make_matrix_view(*task.C, task.dim, task.dim);

const auto A_block = std::submdspan(
	A_view,
	std::pair{ task.row_begin, task.row_end },
	std::full_extent);
auto C_block = std::submdspan(
	C_view,
	std::pair{ task.row_begin, task.row_end },
	std::full_extent);

gemm(A_block, B_view, C_block);
```

数组生成程序将随机整数以二进制 `int32_t` 记录，每个元素固定占用 4 字节。

```cpp
std::ofstream out(path, std::ios::binary);
std::uniform_int_distribution<std::int32_t> u(
	std::numeric_limits<std::int32_t>::min(),
	std::numeric_limits<std::int32_t>::max());

for (std::size_t i = 0; i < size; ++i) {
	const std::int32_t value = u(e);
	out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
```

数组求和 worker 各自打开同一输入文件并定位到自己的起始记录，顺序读取自己负责的元素区间，将 `int32_t` 分块读入局部缓冲区并累加。达到固定块长后，局部块和被写入线程私有的高精度整数。

```cpp
in.seekg(static_cast<std::streamoff>(task.begin * record_size), std::ios::beg);

std::vector<std::int32_t> buffer(buffer_size);
std::int64_t chunk = 0;
std::size_t count = 0;
std::size_t remaining = task.end - task.begin;
while (remaining > 0) {
	const std::size_t take = std::min(remaining, buffer.size());
	in.read(reinterpret_cast<char*>(buffer.data()), take * record_size);

	for (std::size_t i = 0; i < take; ++i) {
		chunk += buffer[i];
		++count;
		if (count == flush_interval) {
			flush_chunk(task.sum, chunk, count);
		}
	}
	remaining -= take;
}
flush_chunk(task.sum, chunk, count);
```

高精度整数采用 $10^9$ 作为内部进制，以小端顺序保存各段十进制数。加法先按符号判断是执行绝对值加法还是绝对值减法，输出时再从最高段开始转换为字符串。数组求和最终对每个线程的局部高精度和做一次归并。

```cpp
class BigInteger final {
public:
	void add(std::int64_t value);
	void add(const BigInteger& other);
	std::string to_string() const;

private:
	static constexpr std::uint32_t base_ = 1'000'000'000;
	bool negative_ = false;
	std::vector<std::uint32_t> digits_;
};
```


= 实验结果

矩阵乘法的结果如下。

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  align: center + horizon,
  inset: 8pt,
  table.cell(
    rowspan: 2,
    align: center,
  )[
    线程数
  ],
  table.cell(
    colspan: 5,
    align: center + horizon,
  )[
    矩阵规模
  ],
  [128], [256], [512], [1024], [2048],
  [1], [0.000927], [0.003080], [0.019441], [0.155690], [2.292907],
  [2], [0.000759], [0.001878], [0.010916], [0.079265], [1.130514],
  [4], [0.000737], [0.001303], [0.005932], [0.043111], [0.641801],
  [8], [0.000947], [0.001301], [0.005058], [0.029651], [0.449295],
  [16], [0.001043], [0.001669], [0.004451], [0.020301], [0.244978],
)

矩阵规模较小时，不同线程数之间差别不明显，有时线程数增加后耗时反而略增。随着矩阵规模增大，多线程版本的耗时整体呈下降趋势，说明按输出矩阵的行块进行划分能够在较大计算量下体现出并行效果。与 lab2 中的 MPI 矩阵乘法相比，使用 Pthreads 的版本在较大矩阵规模下能够获得更好的性能，这可能与线程间共享内存的低通信开销有关。

数组求和的结果如下。

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  align: center + horizon,
  inset: 8pt,
  table.cell(
    rowspan: 2,
    align: center,
  )[
    线程数
  ],
  table.cell(
    colspan: 5,
    align: center + horizon,
  )[
    数组规模
  ],
  [1M], [4M], [16M], [64M], [128M],
  [1], [0.001968], [0.004411], [0.013120], [0.046469], [0.091005],
  [2], [0.001935], [0.003177], [0.008840], [0.028104], [0.053515],
  [4], [0.002124], [0.002944], [0.006310], [0.019060], [0.035768],
  [8], [0.002776], [0.003428], [0.006439], [0.016211], [0.031391],
  [16], [0.004840], [0.005163], [0.008409], [0.019407], [0.033284],
)

耗时的变化趋势与矩阵乘法基本一致。

= 实验感想

本实验完成了基于 Pthreads 的矩阵乘法和数组求和实现，并使用统一命令行接口完成了不同规模、不同线程数下的批量评测。实验结果表明，任务划分方式需要与计算密度和输入方式相匹配；矩阵乘法适合按输出行块并行计算，数组求和则可以利用二进制定长文件格式实现线程独立定位与读取，从而在不整体载入数组的前提下完成并行聚合。
