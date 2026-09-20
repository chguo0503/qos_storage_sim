# 输入文档审查：原样比较ASU与OD

审查对象：[continuous_underload_reproduction.md](../../docs/continuous_underload_reproduction.md)。按用户最新要求，本实验保留文档的32 NPU、3 SSU、每盘40 GiB/s、10种原始data画像、每卡每画像2条、Random/Ordered顺序；不做错峰、不提高容量、不改计算时间，也不删除超过40 GiB/s的时段。是否发生实际超限由后续完整仿真逐事件报告，不能预先把目录名或“欠载”标题当成验证结论。

## 1. 不能称为旧归档的逐条复现

当前`diverse_data_ssu3_l3_20260916/inputs`包含semi的3份960请求输入、full的3份1344请求输入，以及1份2304请求full_guaranteed输入。该目录35份运行manifest由16份960请求、16份1344请求、3份32请求smoke组成，未找到640请求的under归档。引入该研究的提交`1c2bb30`也没有这组归档。

这些历史manifest写明`layout=stripe_npu_mod_ssu`，块落盘规则为`(block_index+npu_id)%3`，并非Ring hash。当前生成器后来改成Ring hash，不能据此把旧运行也说成Ring hash。本次准确定位为：**依照文档画像和队列规则新建640请求，再用当前Ring hash比较ASU与OD**。

## 2. 原始画像与层数

10个画像key均存在于项目根目录的`data`。每项是4元组`(B, C_us, source_TTFT_ms, V_GiB)`，没有独立的层数字段。其数值满足`B=V/(C_us/1e6)`，源TTFT字段满足`source_TTFT_ms=78*C_us/1000`。

| 总输入长度 | miss token | 每层读取MiB | 每层计算ms | B GiB/s |
|---:|---:|---:|---:|---:|
| 32K | 2048 | 41.25 | 14.369103 | 2.803460 |
| 32K | 4096 | 38.50 | 28.592842 | 1.314932 |
| 64K | 2048 | 85.25 | 25.106518 | 3.315950 |
| 64K | 4096 | 82.50 | 50.067670 | 1.609150 |
| 80K | 2048 | 107.25 | 30.475233 | 3.436769 |
| 80K | 4096 | 104.50 | 60.805103 | 1.678326 |
| 128K | 2048 | 173.25 | 46.581359 | 3.632128 |
| 128K | 4096 | 170.50 | 93.017325 | 1.790031 |
| 160K | 2048 | 217.25 | 57.318773 | 3.701374 |
| 160K | 4096 | 214.50 | 114.492172 | 1.829581 |

本次明确使用8层，不把源TTFT字段直接用作8层SLO基线。请求纯计算时间为`8*C`，SLO×1.5的门槛为`1.5*8*C`。20请求的每卡纯计算总时间约8.333218秒。所有640条请求在0秒到达，保持有限输入；Random随机的是每卡内部的顺序，不是到达时刻。读数应区分接纳起算与到达起算，实际首token事件仍未建模。

## 3. 只修正hash身份，不改变文档队列顺序

文档代码先shuffle、再用队列位置编号。若直接用这个编号作为Ring hash key，同一具体请求在Random与Ordered中可能换到不同盘，比较就不再只改变顺序。

为保持物理请求集合一致，先按文档canonical顺序给每卡20个具体请求固定编号，然后才shuffle：

```text
canonical顺序：长度递增，同长度miss递增，每个画像的两个副本相邻
stable_identity = npu_id * 20 + canonical_ordinal
runtime_request_id = npu_id * 20 + queue_position
Random：Random(seed + npu_id).shuffle(20个具体请求)
Ordered：保持canonical顺序
落盘：sim.block_ring_hash_disk_id(stable_identity, block_index, 3)
```

稳定身份存入`load.original_request_id`。对20个具体请求shuffle与文档直接shuffle20个画像tuple产生**完全相同的画像队列序列**；这里只分离“请求身份”和“执行位置”，不改变Random规则或Ordered排序。所有策略复用冻结manifest，不各自生成新请求。

## 4. 数学上不能保证Ring hash逐盘持续欠载

对每张卡的20条固定请求，先取其在盘s上最大的每层需求，再把32卡相加：

```text
H_s = sum_i max_request_on_NPU_i(V_request,s / C_request)
任意时刻的当前请求名义需求 D_s(t) <= H_s
若每盘H_s<40，可直接证明任何画像组合都欠载
若H_s>40，只代表这个充分条件不成立，不能替代实际运行审计
```

按上述连续稳定身份、640个具体请求和当前Ring hash计算：

| 指标 | 盘0 GiB/s | 盘1 GiB/s | 盘2 GiB/s |
|---|---:|---:|---:|
| Ring hash的静态上界H_s | 38.805665 | 42.334935 | 41.292819 |
| 按每卡全部纯计算时间加权的理论平均需求 | 23.358137 | 25.606293 | 25.121599 |
| 假如使用旧stripe的静态上界（仅作数学对照） | 39.482297 | 39.482297 | 39.479368 |

平均需求总和约74.086029 GiB/s，低于120，但这不能证明每盘每时刻低于40。旧stripe的均匀条带能给出约39.48的静态保证，当前Ring hash的盘间不均衡使该保证失效。**本次不恢复stripe，也不修改这些输入以强行通过欠载检查。**

还可以算出同步画像组合的例子：

| 所有32卡当前画像及副本 | 盘0 GiB/s | 盘1 GiB/s | 盘2 GiB/s |
|---|---:|---:|---:|
| 128K/2048，第1副本 | 36.310469 | 40.501109 | 39.416515 |
| 128K/2048，第2副本 | 37.135625 | 39.477771 | 39.614697 |
| 160K/2048，第1副本 | 37.672606 | 40.712183 | 40.059172 |
| 160K/2048，第2副本 | 37.327066 | 41.101648 | 40.015248 |

因此Ordered同步推进到这些画像时存在逐盘超限风险。表格描述的是给定画像组合的数学需求，不是已经观测到的执行时序；各卡实际偏移、等待和策略差异要等仿真后确认。Random同样按完整执行轨迹审计，不通过删掉超限时段制造“持续欠载”。

## 5. 后续结果需要怎样读

ASU在每盘共用Path0。OD每盘给每NPU独立Path，CIR=40/32=1.25 GiB/s，PIR不设硬限，按现有组间和组内WRR借用空闲份额。OD的CIR均分不是瞬时实际供给均分，也不代表一定改善每个请求的SLO。

实际输出应分别展示ASU/OD、Random/Ordered的利用率和SLO，并报告逐盘超限占比、最大需求及发生区间。名义需求按当前已接纳请求的`V_s/C`统计，不重复加上下一请求接纳前首层预取；这是实验的参考需求口径，不是瞬时I/O流量绝不超过容量的声明。实际供给另从SSU服务事件统计。

数学审查不预填策略性能结论。即使输入在某段超限，也保留该段及原标签，把“是否严格欠载”和“OD表现如何”作为两个分别报告的问题。

## 6. 冻结输入生成后的只读复核

[verify_input_design.py](verify_input_design.py)只读取已有manifest和原始data，打印JSON，不生成输入、不运行仿真、不改文件。`--manifest`可重复，复核不同顺序/种子使用相同物理请求集合：

```bash
python results/continuous_underload_asu_od_20260918/verify_input_design.py \
  --manifest results/continuous_underload_asu_od_20260918/inputs/random_seed7_ring_hash.json.gz \
  --manifest results/continuous_underload_asu_od_20260918/inputs/ordered_seed7_ring_hash.json.gz
```

它检查640/20/2配比、10种原始画像、8层、全部arrival=0、文档Random队列、两类ID、逐块Ring hash和同一物理请求集合，并重算本页静态界与同步画像例子。输入SHA在读取前后保持不变。实际超限和性能仍由正式运行结果决定。

已对生成后的Random/Ordered、seed 7/19/43共6份冻结manifest实际执行该复核，全部通过：640个具体请求的画像、物理落盘和计算参数一致，Random画像序列与文档代码逐项相同，6份输入重算出的静态界均与上表相同。该检查没有运行仿真。
