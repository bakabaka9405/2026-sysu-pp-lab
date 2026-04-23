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
  [课程名称：并行程序设计与算法],
  align(right)[批改人：#h(12em)],
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
  [实验], [], [专业（方向）], [计算机科学与技术（人工智能与大数据）],
[学号], [#id], [姓名], [#author],
[Email], [#text(size:10pt, email)], [完成日期], [2026年4月3日],
)

#v(1em)

#let section-title(num, title) = {
  text(size: 16pt, weight: "bold")[#num. #title]
}

= 实验目的


= 实验过程和核心代码

== 实验平台配置

- 操作系统：Windows 11 23H2 22631.6199
- 处理器：13th Gen Intel(R) Core(TM) i5-13600K
- Microsoft MPI 版本：10.1.12498.18



= 实验结果



= 实验感想
