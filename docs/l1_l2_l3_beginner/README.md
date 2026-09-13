# L1 / L2 / L3：手稿与实验的零基础教程

**重点看 b_i、B_i：[周六实验对齐说明](周六实验_bi与Bi对齐说明.md) · [局部图 PNG](assets/bi_Bi_A_local.png) · [局部图 PDF](assets/bi_Bi_A_local.pdf)。** 在同一真实 A 周期中，对齐 NPU 到达带宽、参考需求、累计字节与计算/等待，解释为什么整轮 `5 / 28.48 = 17.56%` 能与实测相符，以及为什么不能套用到整个混合窗口。运行 `python docs/l1_l2_l3_beginner/build_bi_alignment.py` 重建；使用逐块接收事件，无新仿真，来源及校验见 `bi_alignment_checks.json`。

**已读教程后，推荐：[周六实验与 NPU 利用率：四页核心版 PDF](周六实验与NPU利用率_四页核心版.pdf) · [HTML](周六实验与NPU利用率_四页核心版.html)。** 结合 2026-09-12 的 A/B 实验，只保留输入与结果、真实逐层等待、三个纯文本公式、调度含义及证据边界。表格采用 Random 三种子均值，局部图明确标为 seed 7；没有新增实验。运行 `python docs/l1_l2_l3_beginner/build_core_digest.py` 可重建，数据与版面核验保存在 `core_digest_checks.json`。

**推荐阅读新版：[手稿融合 PDF](L1_L2_L3_手稿与实验_零基础融合教程.pdf) · [新版 HTML](L1_L2_L3_手稿与实验_零基础融合教程.html)。**

补充阅读：[两个利用率公式的关系：一个看“现在”，一个估计“平均”](两个利用率公式的关系_通俗说明.md)。使用纯文本公式和同一张卡的计算／等待例子，解释 `U_est(t)`、`U_actual(t)`、窗口平均以及 `t` 与 `T`。

2026-09-13 修订：[第 23 页改用逐层预取、相同窗口的例子](第23页修订说明_layerwise.md)，替换原单层先读后算与不同全程时长的例子。两卡各执行 8 层，在同一 `[6,12)` 秒窗口内比较层 IO 优先级；这是经过时序核验的过载教学构造，不是新增 Baseline/Once 实验。

新版 35 页，一页推进一个问题，把两张 JPG 的 36 个公式/结论条目融合进讲述过程：先解释求和、平均、倒数、min/max 和积分，再逐等号解释手稿，并对应实际实验。每项附条件或纠正；数学练习使用小数字，原稿个别字迹不清处明确标注。35 个 PDF 书签可直接定位。

构建新版：

```bash
python docs/l1_l2_l3_beginner/build_actual_figures.py
python docs/l1_l2_l3_beginner/build_math_figures.py
python docs/l1_l2_l3_beginner/build_fused_guide.py
```

`manuscript_inventory.json` 保留逐条转写、条件、小例和字迹疑点；`manuscript_coverage.json` 对应新版页码。`fused_page_content.json` 为正文，`fused_layout_audit.json` 和 `fused_pdf_checks.json` 保存布局、页数、书签及哈希核查。没有新增仿真。

以下保留 18 页初版及其构建说明。

[阅读 PDF](L1_L2_L3_NPU_小白图解.pdf) · [浏览 HTML](L1_L2_L3_NPU_小白图解.html)

18 页入门重写版，从一张卡的计算/等待开始，再讲 A/B、预取、三级调度、真实实验、自然错开、有限人口、开环闭环、Once 与 TTFT。第 17–18 页是可选的公式速查。原教程保留不覆盖。

本次没有启动新仿真。主结果使用 `baseline_ab128_32_ratio12_20260912` 的长验证 seed7；暖窗、长窗、全程和局部片段分别标注。教学图与真实日志图明确区分。

在项目根目录重新构建：

```bash
python docs/l1_l2_l3_beginner/build_actual_figures.py
python docs/l1_l2_l3_beginner/build_guide.py
```

使用环境已有的 Python、matplotlib、pypdf、Google Chrome 和中文字体。HTML 排版先检查正文与页脚不重叠，再生成 PDF，核验页数和逐页文本，并添加 18 个阅读书签。

- `assets/`：四张独立真实时序图的 SVG、PNG，及来源审计。
- `page_content.json`：逐页标题、正文和来源。
- `source_audit.json`、`layout_audit.json`、`pdf_checks.json`：来源哈希、布局及 PDF 核验。
- [详细公式核对](../../results/baseline_ab128_32_ratio12_20260912/formula_review/README.md)：全部公式编号及精确数字。
