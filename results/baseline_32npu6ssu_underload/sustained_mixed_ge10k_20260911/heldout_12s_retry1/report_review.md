# build_report.py 有界只读科学审阅

审阅版本 SHA256：`4049b3e8376d906520f23ecd6e8a8e6cc7ba9e60962c1d66a6ec516e988762e8`。未运行 builder、绘图器或任何仿真；未改 root 文件。本文是在 root 已补充实际快短份额、子窗混合表和预取代理表之后，对最新稿的复核。

已核正确：主人口 20,032 与 frozen plan 相符；6L/48S 两组循环和初始移位准确；29 个 2 秒窗的 U 汇总按实际计算时间计算，SLO 按接纳人口计数合并，未把比例直接取算术均值；画像 compute/active 与整机 compute/(32T) 分母有明确区分；欠载证书 39.898850 GiB/s 正确；heldout 主 U/SLO/25 卡子窗数字均与独立审计一致；没有把两提交种子写成独立业务样本。60 秒数值尚未生成，不对未完成格作结果判断。

## 需要修正

1. **新混合卡数表仍取了含 bridge 的原 role 字段，不能配合快短定义解释。** 当前 L51 的 `mixed_card_count` 来自 `audit_sustained.window`：Short 包含 NQL4096 bridge。L91 又把快短限定为 NQL1024，读者自然会把“三列两类卡数”理解为快短+Long。尤其 Random 的 2 秒窗口可能包含 bridge 和 Long，却没有快短，两种卡数可不同。建议主表三列均使用 `sum(c['fast_short_C_ms'] > 0 and c['roles']['long']['compute_ms'] > 0 for c in window['per_npu'])`，表头写“快短+Long”；原 role 卡数若要保留，单独标注“Short含bridge”。`draw_final_timeline.collect` 已使用真正的 fast+Long，报告需与其一致。

2. **原筛选通过标记与快短≥5%要分开。** L68 的 `primary_conditions_passed` 仍验证原 role（bridge 算 Short）≥5%，不是 `cards_fast_and_long_ge5percent==32`。原筛选定义可以保留，避免事后更换判据；但应在其后另列每个策略真正快短与 Long 各≥5%的卡数。目前实际 min 快短/Long 表已是重要改善。L85 的配额句可更直接改为：“以每卡全部计算（包含bridge）为分母，真实快短的最小份额为5.4963%，Long的最小份额为8.1008%；bridge不进入快短分子。” 若需要原 role 最小8.1008%，另句说明，避免两种 Short 概念来回切换。

3. **L103 对物理原因的断言超出本批层日志能定位的范围。** “等待的是共享SSU队列中排在前方的读取”容易被读成已经逐块证明所有停顿都由SSU FIFO前驱造成；当前只是层级read/compute/stall及相位证据，读取还经过NPU链路。建议：“层日志证明这些短请求在等数据。Baseline各SSU的Path0按FIFO服务，集中长读可能推迟随后短读；但本批未采集逐块trace，不能把每次stall唯一分摊为某个Long前驱或某段盘服务。另一张卡的长计算本身不占用这张卡的计算单元。” 保留源码所支持的机制解释，不把read lifetime认作SSD busy。

4. **L133 ‘一个只有较慢辅助短请求的对照’范围不准确。** `pilot_bridge_control` 的后16卡仍有 S32/S48/S64 三个快短画像；只有前16原长组没有快Short、仅插入bridge。建议改为：“一个原长组仅插入慢bridge、后16卡仍有三种快短的机制对照，约90.1%”。同样建议在主输入说明后加一句：“前16卡的真实快短只有64K/1024一种，后16卡包含三种；bridge单列”，避免把全局多短画像读成每卡都有多种快短。

## 建议补清的短句

- L38 的“辅助预取请求”改为“桥接请求（真实计算）”。它是一条真实请求，有自身8层读取与计算，不是额外的纯预取操作。L78已有C接近Long的限制，保留即可。
- 画像利用率表 L105 前明确“Ordered Baseline、seed7、[2,60)s”，因为该表只取 `ordered['primary_window']`，不是四策略汇总。
- 方法或限制中加：“本表Once是原始Once per layer，5ms快照、固定NPU绑定；不是NewOnce。模拟中控制CPU/通信额外延时设为0，墙钟运行成本不计入设备U。” 这些是现有源配置事实，不是新增实验结论。
- L101 的0.01353ms来自 `bridge16...24s1l` 的10秒pilot，数值由 `notes/bridge_method.md` 支持，未发现错误；可直接加该链接和“短组24S/1L”标签，避免误认为它就是主组48S/1L的60秒结果。

## 可直接引用的 heldout 文案

“预先声明的提交种子19、43复验使用同一份确定队列，只改变同时就绪NPU的提交顺序。Baseline在[2,12)s的设备U分别为89.774233%和89.727954%；十个2秒子窗为89.518178%–89.923987%。所有窗口32卡active、idle=0，完整运行逐盘current V/C峰值36.011188 GiB/s，超过40的时长为零。主窗每卡都有真实快短与Long，快短最低份额5.050376%，Long最低份额6.326273%/6.333125%；bridge不计入快短。每个2秒子窗只有25卡同时覆盖两类，因此不能称任意2秒每卡均混合。”

“前16卡排除各自首次L0后，每格8176层暴露stall均为零；8轮计算起点跨度保持约0.016884668/0.016584668ms。其S64最大read lifetime为6.916858/6.885585ms，最小deadline余量5.709083/5.740356ms。这支持相位被保留的解释，但不是某个FIFO物理前驱的逐块证明，也不是无限期保证。”

首次heldout未知SIGTERM没有完整结果，原记录保留；一次授权重试后2/2完整通过，不是筛掉表现差的seed。此执行说明在当前稿中已正确。

## 同时只读核对最终四色图

`draw_final_timeline.collect` 正确把 NQL1024 的fast、32K/NQL4096的bridge、Long分别绘制；三色C、fast+bridge与原Short C、stall都逐卡对齐原时间账；fast份额分母是全部该卡C。`base.collect` 先断言全32 active，支撑图中无idle说明。没有发现阻塞性的分色或时间守恒错误。未运行或修改该绘图器。
