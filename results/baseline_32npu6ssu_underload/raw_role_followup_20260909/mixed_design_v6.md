# v6：提前耗尽Guard的Long，给快速Short留下交接间隔

冻结模块[mixed_design_v6.py](mixed_design_v6.py)，SHA-256 `9faa9e1942ec598dfe1e2a1a811f14a8a7dce00e831666811de33be8f7a275b8`。API为`build_queues(source_requests, source_metadata, seed, mode)`，random/ordered生成后仍返回原对象，由wrapper明确重绑NPU与位置ID。

原始raw200固定来源的1172条请求不变：Long200K/1024共380条，Short32/48/64K、NQL1024各264条。v6重新分配逐卡人口；排序的准确对照是这套新绑定的random。

| 卡号 | 每种Short数S1/S2/S3 | Long数 | Ordered |
|---|---|---:|---|
| 0–16 | 3/3/3 | 13 | 全部L→全部Short轮转 |
| 17–19 | 8/8/8 | 8 | 全部L→全部Short轮转 |
| 20–22 | 16/16/16 | 12 | 下述后12卡顺序 |
| 23–28 | 16/16/16 | 11 | 同上 |
| 29–31 | 15/15/15 | 11 | 同上 |

后12卡是独立洗牌9S2+8S3→连续15S1→全部L→剩余Short轮转。其**观察窗附近的重点短段重复同一S1画像**，虽然完整队列含三种Short；这一人为集中必须披露，不能称为整个热段独立多画像混合。v7单独改变热段来检查此限制。Short实际分类均为SL，Long为LL。

Guard的8L纯C2268.348860004ms；后12卡在首次Long前必须完成2394.722223208ms纯C的Short，差126.373363204ms。每卡完整纯C最少4177.133423111ms。五seed真实输入的random/ordered共160个逐卡配对，身份、准确配额、C/V/placement原对象、确定性检查均通过，见[mixed_design_v6_checks.json](mixed_design_v6_checks.json)。

## Baseline ordered的FIFO充分条件

以下是针对本模拟器的分析推导，已由两个独立审阅者只读核过关键语义，不是Once的服务保证，也不是设备U下界。数值可用[mixed_design_v6_fifo_bound.py](mixed_design_v6_fifo_bound.py)从真实source复算；[JSON](mixed_design_v6_fifo_bound.json)保存输入、生成器和核心源码SHA。

条件是batch=1、普通单层预取、全部I/O进入每盘Path0 FIFO、每盘40GiB/s、NPU接收50GiB/s、176KiB块、每块0.1µs提交、无额外人为延迟。`continuous_batch_sim.py:3270`等处要求当前层全部`io_ready`才计算；计算开始时只启动下一层，或在末层启动下一请求L0。因此每卡最多有一层尚未完成的传输。`sim.py:580`起为Path的FIFO队列；`sim.py:1128`起按盘带宽非抢占地服务命令；`sim.py:105`和`:113`给出SSD、接收链路的纯字节服务时间。

在后12卡第一次Long L0之前，它们的未完成传输仍来自Short，其余20卡每层最大是Long。Long每盘最多266块，Short最多84块。在被分析的某个Long层**全部块已提交的时刻**，任一盘当前所有残余传输工作（含active残余）不超过：

`(20×266 + 12×84) × 176KiB = 1.0621337890625GiB`。

这是将所有未完成层按完整体积计入的**保守上界**。Path0 FIFO下，后来提交的块不能超过该层已提交的最后一块，因此其SSD余下等待加服务最多26.553344727ms。全层最多1592块，总提交间隔上界0.1592ms包含上一轮`client_next_issue`的至多一个间隔残余；再保守地计入本卡整个Long层经50GiB/s接收所需5.344238281ms，得到release→ready上界：

`0.1592 + 26.553344727 + 5.344238281 = 32.056783008ms`。

它小于Long一层C35.442950938ms。首次L0也用相同上界，随后各层及跨Long请求L0可归纳完全隐藏。所以Guard耗尽8L的时刻上界约为：

`8×8×35.442950938 + 32.056783008 = 2300.405643012ms`。

后12卡首次Long L0是在最后一条S1的L7**计算开始**时释放，其仅由纯C得到的最早下界是`2394.722223208 − 7.257231579 = 2387.464991629ms`，比Guard上界晚87.059349ms，能闭合此前的角色条件。数值中的微小浮点事件epsilon远小于这个间隔；实际结果仍须独立事件核查。

Guard三张卡耗尽所有Long后不再回Long，因此Baseline ordered在此模型下，全程当前Long卡数≤29；任意盘名义需求≤39.881116880GiB/s。这个证明不说明短请求deadline全部可满足，也不保证后12卡在4秒前开始Long或整机U低于某值。

## 不推广到Once或random

Once使用多个Path，后到其他Path的块可能先于某个Long末块服务，不能直接用上述Path0 FIFO工作界。v6扩大了时间间隔，但Once的Guard退出、全程逐盘名义需求仍要实际扫描；random也不服从上述分段顺序。所有策略都须分别核暖窗每卡至少100ms Long/Short正计算、全窗active、角色并发、设备U及暖窗admission处理SLO，失败seed保留。静态任意组合上界仍40.310157583GiB/s，不能称整套画像在任何排序下都已证明欠载。
