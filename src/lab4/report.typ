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
  [实验], [Pthreads并行方程求解及蒙特卡洛], [专业（方向）], [计算机科学与技术（人工智能与大数据）],
  [学号], [#id], [姓名], [#author],
  [Email], [#text(size: 10pt, email)], [完成日期], [2026年5月7日],
)

#v(1em)

#let section-title(num, title) = {
  text(size: 16pt, weight: "bold")[#num. #title]
}

= 实验目的

== 一元二次方程求解
使用 `Pthread` 编写多线程程序，求解一元二次方程组的根，结合数据及任务之间的依赖关系，及实验计时，分析其性能。

*一元二次方程*：为包含一个未知项，且未知项最高次数为二的整式方程式，常写作 $a x^2+b x+c=0$，其中 $x$ 为未知项，$a,b,c$ 为三个常数。

*一元二次方程的解*：一元二次方程的解可由求根公式给出：

$ x = (-b plus.minus sqrt(b^2 - 4 a c)) / (2 a) $

输入：$a,b,c$ 三个浮点数，其的取值范围均为 $[-100, 100]$

问题描述：使用求根公式并行求解一元二次方程 $a x^2+b x+c=0$。

输出：方程的解 $x_1,x_2$，及求解所消耗的时间 $t$。

要求：使用 `Pthreads` 编写多线程程序，根据求根公式求解一元二次方程。求根公式的中间值由不同线程计算，并使用条件变量识别何时线程完成了所需计算。讨论其并行性能。

== 蒙特卡洛方法求 $π$ 的近似值

基于 `Pthreads` 编写多线程程序，使用蒙特卡洛方法求圆周率 $π$ 近似值。

蒙特卡洛方法与圆周率近似：蒙特卡洛方法是一种基于随机采样的数值计算方法，通过模拟随机时间的发生，来解决各类数学、物理和工程上的问题，尤其是直接解析解决困难或无法求解的问题。其基本思想是：当问题的确切解析解难以获得时，可以通过随机采样的方式，生成大量的模拟数据，然后利用这些数据的统计特性来近似求解问题。在计算圆周率 $π$ 值时，可以随机地将点撒在一个正方形内。当点足够多时，总采样点数量与落在内切圆内采样点数量的比例将趋近于 $π/4$，可据此来估计 $π$ 的值。

输入：整数 $n$，取值范围为 $[1024, 65536]$

问题描述：随机生成正方形内的 $n$ 个采样点，并据此估算 $π$ 的值。

输出：总点数 $n$，落在内切圆内点数 $m$，估算的 $π$ 值，及消耗的时间 $t$。

要求：基于 `Pthreads` 编写多线程程序，使用蒙特卡洛方法求圆周率 $π$ 近似值。讨论程序并行性能。


= 实验过程和核心代码

== 实验平台配置

- 操作系统：Windows 11 23H2 22631.6199
- 处理器：13th Gen Intel(R) Core(TM) i5-13600K 12 核 20 线程
- GCC 版本：gcc version 16.1.0 (Rev1, Built by MSYS2 project)

== 实验过程

=== 一元二次方程求解

一元二次方程求解的核心计算流程可进一步细化为四个阶段：第一阶段并行计算 $b^2$ 与 $4 a c$；第二阶段将二者相减得到判别式 $D=b^2-4 a c$；第三阶段对 $D$ 开平方得 $sqrt(D)$；第四阶段并行计算两个根 $x_1=(-b+sqrt(D))/(2a)$ 与 $x_2=(-b-sqrt(D))/(2a)$。四个阶段之间存在明确的数据依赖：第二阶段依赖第一阶段的两个结果就绪，第三阶段依赖第二阶段的结果就绪，第四阶段依赖第三阶段的结果就绪，因此适合使用条件变量构建多层级流水线同步网络。

流水线的实现基于 `Pthreads` 的条件变量机制。定义一个共享上下文结构体 `PipelineCtx`，保存方程系数、六个中间结果、一个互斥锁和五个条件变量，分别用于标记各阶段计算完成。每个条件变量对应一个布尔标志位，供等待线程在 `pthread_cond_wait` 返回后做二次检查，以避免虚假唤醒。

```cpp
struct PipelineCtx final {
	double a = 0.0;
	double b = 0.0;
	double c = 0.0;

	double b_sq = 0.0;
	double ac_4 = 0.0;

	double D = 0.0;
	double sqrt_D = 0.0;
	double x1 = 0.0;
	double x2 = 0.0;

	pthread_mutex_t mutex{};
	pthread_cond_t cond_bsq{};
	pthread_cond_t cond_4ac{};
	pthread_cond_t cond_D{};
	pthread_cond_t cond_sqrt{};
	pthread_cond_t cond_done{};

	bool bsq_ready = false;
	bool ac4_ready = false;
	bool D_ready = false;
	bool sqrt_ready = false;
	bool x1_ready = false;
	bool x2_ready = false;

	std::exception_ptr error;
};
```

六个工作线程分别对应流水线的四个阶段，其中第一阶段和第四阶段各包含两个并行线程，第二阶段和第三阶段各包含一个线程。

第一阶段并行启动 `calc_bsq` 和 `calc_4ac` 两个线程。它们各自获取互斥锁后执行单一乘法运算，前者计算 $b^2$，后者计算 $4 a c$，计算完成后将对应标志位置为 `true` 后通过 `pthread_cond_signal` 通知下游，最后释放互斥锁。两个线程之间无依赖关系，可并行执行。

```cpp
void* calc_bsq(void* arg) {
	auto& ctx = *static_cast<PipelineCtx*>(arg);
	pthread_mutex_lock(&ctx.mutex);
	ctx.b_sq = ctx.b * ctx.b;
	ctx.bsq_ready = true;
	pthread_cond_signal(&ctx.cond_bsq);
	pthread_mutex_unlock(&ctx.mutex);
	return nullptr;
}
```

`calc_4ac` 的结构与 `calc_bsq` 对称，区别仅在于计算 $4 a c$ 并使用 `cond_4ac` 进行通知。

第二阶段由 `calc_D` 线程完成，它需要等待两个独立的前置条件同时满足。`calc_D` 进入临界区后，先以 `while (!bsq_ready)` 循环等待 `cond_bsq`，被唤醒且确认 $b^2$ 已就绪后，再以 `while (!ac4_ready)` 循环等待 `cond_4ac`。每次被唤醒后均首先检查 `error` 字段，若上游线程已发生异常则立即设置 `D_ready` 并传播错误信号。只有当两个前置结果均已就绪且无异常时，才计算 $D = b^2 - 4 a c$。

```cpp
void* calc_D(void* arg) {
	auto& ctx = *static_cast<PipelineCtx*>(arg);
	pthread_mutex_lock(&ctx.mutex);

	while (!ctx.bsq_ready) {
		pthread_cond_wait(&ctx.cond_bsq, &ctx.mutex);
	}

	while (!ctx.ac4_ready) {
		pthread_cond_wait(&ctx.cond_4ac, &ctx.mutex);
	}

	ctx.D = ctx.b_sq - ctx.ac_4;
	ctx.D_ready = true;
	pthread_cond_signal(&ctx.cond_D);
	pthread_mutex_unlock(&ctx.mutex);
	return nullptr;
}
```

第三阶段由 `calc_sqrtD` 线程完成。该线程等待 `cond_D` 后调用 `std::sqrt` 计算平方根，并置 `sqrt_ready` 为真。与之前设计的关键差异在于，此处使用 `pthread_cond_broadcast` 而非 `pthread_cond_signal` 来通知下游——因为第四阶段有两个线程同时等待 `cond_sqrt`，必须确保两者均被唤醒。

```cpp
void* calc_sqrtD(void* arg) {
	auto& ctx = *static_cast<PipelineCtx*>(arg);
	pthread_mutex_lock(&ctx.mutex);

	while (!ctx.D_ready) {
		pthread_cond_wait(&ctx.cond_D, &ctx.mutex);
	}
	ctx.sqrt_D = std::sqrt(ctx.D);
	ctx.sqrt_ready = true;
	pthread_cond_broadcast(&ctx.cond_sqrt);
	pthread_mutex_unlock(&ctx.mutex);

	return nullptr;
}
```

第四阶段由 `calc_x1` 与 `calc_x2` 两个线程并行完成。两者均等待 `cond_sqrt`，被唤醒且确认 $sqrt(D)$ 已就绪后，分别计算 $x_1 = (-b+sqrt(D))/(2a)$ 和 $x_2 = (-b-sqrt(D))/(2a)$。计算完成后各自置相应的标志位并通过 `cond_done` 做广播通知主线程。

```cpp
void* calc_x1(void* arg) {
	auto& ctx = *static_cast<PipelineCtx*>(arg);
	pthread_mutex_lock(&ctx.mutex);

	while (!ctx.sqrt_ready) {
		pthread_cond_wait(&ctx.cond_sqrt, &ctx.mutex);
	}
	ctx.x1 = (-ctx.b + ctx.sqrt_D) / (2.0 * ctx.a);
	ctx.x1_ready = true;
	pthread_cond_broadcast(&ctx.cond_done);
	pthread_mutex_unlock(&ctx.mutex);

	return nullptr;
}
```

主函数首先从标准输入读取 $a,b,c$ 三个浮点数，随即执行单线程 baseline 计算，直接将求根公式的四个阶段顺序执行并记录耗时。随后创建六个线程并 `pthread_join` 等待全部完成，计时区间从 `pthread_create` 到所有 `pthread_join` 返回。程序向标准输出打印两行共六个空格分隔的数值：单线程的两个根与耗时、多线程流水线的两个根与耗时。

=== 蒙特卡洛方法求 $pi$

蒙特卡洛 $pi$ 估算的并行化策略与流水线不同，各线程之间不存在数据依赖，每个线程可以独立生成随机点并统计落入单位圆内的数量，待所有线程结束后由主线程汇总。

每个线程的工作由 `ThreadTask` 结构体描述，其中包含该线程需生成的采样点数量 `count`、独立的随机数种子，以及本线程统计的圆内点计数。采样点数按整数比例分配——`n / t` 均匀分配到各线程，余数分配给前若干个线程，确保各线程负载差值不超过一个点。

```cpp
struct ThreadTask final {
	std::size_t count;
	unsigned seed;
	std::size_t inside_count = 0;
	std::exception_ptr error;
};

void* worker(void* arg) {
	auto& task = *static_cast<ThreadTask*>(arg);
	std::mt19937 rng(task.seed);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	std::size_t count = 0;
	for (std::size_t i = 0; i < task.count; ++i) {
		const double x = dist(rng), y = dist(rng);
		if (x * x + y * y <= 1.0) ++count;
	}
	task.inside_count = count;
	return nullptr;
}
```

主函数通过命令行参数接收总采样点数 $n$ 和线程数，创建 `ThreadTask` 数组并分配各线程的起止索引与种子，然后依次创建线程、等待完成、累加圆内点计数。计时区间同样覆盖从线程创建到全部线程完成的完整过程。最终向标准输出打印一行四个空格分隔的数值：$n$、圆内点数 $m$、$pi$ 估计值 $4 dot m/n$，以及耗时。

=== 测试方法

任务一的测试使用五组预先构造的系数作为输入。统计每组输入下单线程和多线程版本的输出结果与耗时，并分析两者的差异。

任务二的测试依次在二十个配置点上进行：取 $n$ 分别为 $2^10$、$2^12$、$2^14$、$2^16$，每个 $n$ 下分别以线程数 $1$、$2$、$4$、$8$、$16$ 运行程序，记录每次输出的 $pi$ 估计值与耗时，并计算每个 $n$ 下五个线程数对应的 $pi$ 估计值的均值与方差。

= 实验结果

=== 一元二次方程求解

对五组随机生成的方程系数，分别以单线程和多线程流水线求解，结果汇总如下。

#table(
  columns: (0.4fr, 0.5fr, 0.5fr, 0.5fr, 0.6fr, 0.6fr, 0.8fr, 0.8fr),
  align: center + horizon,
  inset: 6pt,
  table.cell(
    rowspan: 2,
    align: center,
  )[序号],
  table.cell(
    colspan: 3,
    align: center + horizon,
  )[输入],
  table.cell(
    colspan: 2,
    align: center + horizon,
  )[答案],
  table.cell(
    colspan: 2,
    align: center + horizon,
  )[用时 (s)],
  [$a$], [$b$], [$c$], [$x_1$], [$x_2$], [单线程], [多线程],
  [1], [2], [5], [−3], [0.50], [−3.00], [0.000000], [0.000442],
  [2], [3], [−2], [−8], [2.00], [−1.33], [0.000000], [0.000435],
  [3], [−1], [4], [5], [−1.00], [5.00], [0.000000], [0.000446],
  [4], [0.5], [3], [−2], [0.61], [−6.61], [0.000000], [0.000463],
  [5], [10], [−30], [−40], [4.00], [−1.00], [0.000000], [0.000411],
)

从表格可以观察到，单线程版本的五组测试耗时在给定输出精度下均不可分辨，说明一元二次方程求根公式的计算量极低，不足以在微秒精度上产生可分辨的计时差异。多线程流水线版本的耗时则分布在数百微秒的量级,说明对于计算量微小的任务，线程创建、互斥锁获取与释放、条件变量的等待与信号通知所带来的系统调度开销远超过实际计算开销。
=== 蒙特卡洛方法求 $pi$

在四组 $n$ 和五组线程数构成的二十个配置点上，记录了各次运行的 $pi$ 估计值与耗时。耗时结果如下：

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  align: center + horizon,
  inset: 8pt,
  table.cell(
    rowspan: 2,
    align: center,
  )[$n$],
  table.cell(
    colspan: 5,
    align: center + horizon,
  )[线程数],
  [1], [2], [4], [8], [16],
  [1024], [0.000247], [0.000305], [0.000329], [0.000425], [0.000565],
  [4096], [0.000302], [0.000417], [0.000369], [0.000454], [0.000853],
  [16384], [0.000603], [0.000501], [0.000437], [0.000536], [0.000719],
  [65536], [0.001247], [0.000766], [0.000639], [0.000867], [0.000827],
)

耗时数据的趋势和前几次实验基本相同，因此这里不再重复分析。

$pi$ 估计值的结果如下：

#table(
  columns: (0.8fr, 1fr, 1fr, 1fr, 1fr, 1fr, 0.9fr, 1.1fr),
  align: center + horizon,
  inset: 8pt,
  table.cell(
    rowspan: 2,
    align: center,
  )[$n$],
  table.cell(
    colspan: 5,
    align: center + horizon,
  )[线程数],
  table.cell(
    rowspan: 2,
    align: center,
  )[均值],
  table.cell(
    rowspan: 2,
    align: center,
  )[方差],
  [1], [2], [4], [8], [16],
  [1024], [3.164062], [3.039062], [3.136719], [3.164062], [3.210938], [3.14296], [0.0040863],
  [4096], [3.151367], [3.094727], [3.147461], [3.146484], [3.163086], [3.14062], [0.0007018],
  [16384], [3.141602], [3.138916], [3.172852], [3.146973], [3.150635], [3.15019], [0.0001812],
  [65536], [3.150696], [3.140442], [3.133179], [3.152466], [3.136230], [3.14260], [0.0000742],
)

$pi$ 估计值精度与方差的变化与预期相符。

= 实验感想

本次实验实现了两个基于 Pthreads 的并行程序，分别演示了两种截然不同的线程协作模式。对后续并行算法设计提供了宝贵的实践经验。