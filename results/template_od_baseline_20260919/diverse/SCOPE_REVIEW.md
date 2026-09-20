# 模板保留图表与统计口径核查

本文件记录新 OD 绘图前的只读审查。原 `template/qos_experiments_20260919` 中的图、数据和生成器未修改，未执行归档脚本的 `main()`。

## 本轮允许新增的图

只生成 `render_figures.py` 内白名单中的 7 张 PNG：

- full / semi / near35：各一张原 Baseline、原始 Once、OD 的三策略 CDF，保留完整长尾。
- full / semi / near35：各一张新增 OD seed 7 逐 SSU 参考需求、真实 SSD 供给图。
- semi：一张新增 OD seed 7 的全部 32 NPU 计算／等待时序图，上方包含同次 OD 运行的整机需求与供给。

不生成八卡 NQL、`once_8npu`、`fifo_8npu`、`fixed_concurrent_utilization`、`stall_decomposition`、`sensitivity20k_bandwidth_pair` 或同类图。旧模板 index 的 s07、s08 以及 s03 包含这些内容；主 index 整合时应按用户排除范围处理，不能直接把旧 index 全部原样保留为本次展示。

full / semi / near35 的 `NEW_*_profile_shuffle.png` 是冻结输入的每卡队列顺序，不含策略执行时间。它们可注明“ASU / OD / Once 共同输入”；没有必要复制成三份策略图。

## 输入与归档边界

三组使用模板的原始冻结**条带落盘**输入。每组同种子三策略必须使用完全相同的请求、卡号、队列顺序与逐块落盘。2026-09-18 Ring hash 重建实验的 OD 结果不可直接复用。

near35 是每盘理想平均需求约 35 GiB/s、允许局部过载的 24 画像实验，不是此前 10 画像“逐时刻持续欠载”实验。负载分组是输入组的名称；策略改变请求驻留时长后，各盘逐时刻需求分布可以变化。

模板 full 的三份 Once 大型事件结果缺失，但逐请求 CDF 样本和原 CDF 点完整。本轮原 Baseline / Once 曲线从归档样本重新计算，并逐点核对模板原曲线。模板没有保存 full / semi 的逐种子 `comparison.csv` 和 `once_control_comparison.csv`；绘图汇总使用项目中原实验的这两份表，先核对六份对应 manifest 与模板完全相同，再与模板归档的三种子宏观表核对。所有实际读入来源均记录 SHA256。

新增 OD 的带宽和时序只能来自该次 OD 原始结果，不能使用 Baseline 的供给曲线替代。OD 正式仿真完整完成后才生成最终比较图。

## CDF 口径

- `x = (Prefill完成时刻 − 接纳时刻) / (8 × 每层纯计算时间)`。
- 取 `[2,4)` 秒内接纳的请求，并跟踪到最终完成；保留窗后完成及超时请求。
- seed 7、19、43 各自计算 ECDF，再等权平均。不得先合并样本后当作种子等权结果。
- `CDF(1.5)` 对应 SLO×1.5 达标率；不包括接纳前排队，也没有模拟真实首 token 事件。
- 同一完整输入不保证三种策略在 warm 窗口选中同一批请求。
- 原 full / semi CDF 的浮点值保持原样；near35 沿用归档绘图对 1 倍和 1.5 倍边界的 `1e-9 ms` 误差规范化。新 OD 使用同一绝对时间容差，并保留未规范化比值以供审计。

独立重算结果：原 full / semi 的全部 CDF 点误差严格为 0；near35 两条曲线最大误差为 `1.11e-16`。详见 `original_source_audit.json`。

| 原实验 | Baseline SLO×1.5 | 原始 Once SLO×1.5 |
|---|---:|---:|
| full | 40.2954308715% | 76.1302681992% |
| semi | 96.2705497503% | 99.8095238095% |
| near35 | 96.1981504148% | 100% |

这些是模板的旧条带输入结果，不能替换成此前 Ring hash 实验的数值。

## 带宽和时序口径

逐盘需求来自当前已接纳请求的真实逐盘 `V/C` 逐事件总和，在计算与 I/O 等待期间均保留。未接纳下一请求的首层预取不重复叠加为参考需求。SSD 供给按真实服务区间积分，图中使用 10 ms 平均；包含跨请求预取的真实物理读取。near35 的原始 2 ms 格可以精确合并成 10 ms 格。

新图逐盘容量线为 40 GiB/s，整机容量为 120 GiB/s；不混用十进制 GB/s。需求／供给之比不是 NPU 的瞬时利用率。

semi 原 32 卡图的配套说明只读保留，未找到同名图的独立生成器。新增 OD 时序从完整 `microbatch_metrics.layer_metrics` 重建：计算开始前、上一层计算结束后且请求活跃的区间标为等待；不把整个等待区间解释成 SSD 排队，也不把 `io_ready_time_ms` 当作盘完成时间。按裁剪到 `[2,4)` 的真实计算区间重新计算每卡 U，再与 runner 指标比较。

## 归档生成器的副作用

不要为了“只多加 OD”直接运行原始完整报告生成器：

- 原 full / semi `render_threeway_ttft_cdf.py` 的第三策略是固定候选池，不是 OD。
- 两份原 `cdf/render_cdf.py` 的绘图写盘代码位于模块顶层，不能把它们当作无副作用的工具模块直接 import。
- 原 `render_report.py` 的策略列表包含动态 mild / aggressive / static，并会改写报告和其他表格。
- near35 `report_near35.py` 会重写全部三种子的图、多个 CSV、报告、验证文件和 ZIP；本轮不调用。
- 原 `plot_other_saved_manifests.py` 会重写三组输入图和 CSV；本轮共同输入图直接保留。
- 原 8 卡与 E1 `build_plots.py` 会生成本轮明确排除的图，不能作为新增图的批量入口。

因此新 renderer 使用独立白名单和只读来源校验，最终输出 `render_checks.json` 记录图文件集合、SHA256、CDF逐点复核、OD独立统计复核及人工视觉检查状态。
