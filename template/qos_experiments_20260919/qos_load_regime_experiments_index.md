# QoS仿真实验整理：持续过载、局部过载与持续欠载

整理日期：2026-09-19。建议先解压资料包，打开 `index.html` 看图；本文相对链接对应解压后的同名目录。

本次按原始代码、冻结输入、结果文件与CSV重新整理，没有重跑仿真。历史图片原样保留；从已有数据补算的表和补画的随机混排图另作标记。

本包保留持续过载、局部过载、旧AB Random/Ordered、五组公式AB、mixed8及E1等已有材料；欠载随机输入采用near35，sensitivity20k_076加入局部过载目录。需要区分以下归属：

1. **Ordered的71.01% / 71.43% → Once的83.42% / 74.61%，属于A=128K/NQL256、B=32K/NQL4096的旧AB实验。** 该Ordered是每卡`ABB`重复40轮，不是先全部A再全部B，也不是200K/20K组合。
2. **semi与near35是两组独立实验。** 前者Baseline为99.24% / 96.27%，后者为99.25% / 96.20%；画像相同但配额不同，不能按相近数值混成一组。两者都存在局部过载。
3. **欠载随机输入现采用2026年9月16日补跑的near35：每盘平均带宽需求约35 GiB/s，固定配比随机混排。** 它是平均欠载、允许局部过载，不能称为任意时刻都欠载。32 NPU、3 SSU、24种请求画像、每卡42条请求；Baseline的NPU利用率/SLO×1.5为99.25%/96.20%，流量分配为99.40%/100%。原实验目录为`results/diverse_near35_20260916`，包内归入`03_continuous_underload/near35/`。

**目标已按实验ID确认：用户所指AB实验是2026-09-14的 `sensitivity20k_076`，即整机NPU利用率94.6147%→99.2567%的那组。此前按近似带宽值将mixed8列为目标候选是误判，现已纠正。** 该组原始代码、双策略数据及三张配对原图见 [独立入口](02_partial_overload/sensitivity20k_076/README.md) 和第12节。mixed8与精确2048/1024资料保留为其他实验，不作为本目标的替代。

## 1. 实验总表

下表所有箭头均为 **Baseline → 原始Once（图中也称“流量分配”）**，数字单位为%。U均取warm `[2,4)s`；“SLO样本口径”需同时阅读。三种子组均为seed 7/19/43，旧AB为配对seed7。


| 实验 | NPU / SSU | NPU U | SLO×1.5 | SSD带宽利用率 | SLO样本口径 |
|---|---|---|---|---|---|
| 持续过载 full：24画像 | 32 / 3 | 62.38 → 64.44 | 40.30 → 76.13 | 100.00 → 100.00 | warm接纳，三seed等权 |
| 局部过载 semi：24画像 | 32 / 3 | 99.24 → 99.28 | 96.27 → 99.81 | 82.27 → 82.52 | warm接纳，三seed等权 |
| 局部过载补充AB：sensitivity20k_076 | 8 / 1 | 94.61 → 99.26 | 98.44 → 100.00 | 75.05 → 78.45 | warm接纳，seed7；需求定义见第12节 |
| 旧AB 128K/32K Random | 32 / 3 | 90.68 → 91.24 | 77.42 → 79.59 | 92.61 → 未核得 | warm接纳，seed7 |
| 旧AB 128K/32K Ordered | 32 / 3 | 71.01 → 83.42 | 71.43 → 74.61 | 60.26 → 未核得 | warm接纳，seed7 |
| 欠载随机输入 near35：平均欠载、允许局部过载 | 32 / 3 | 99.25 → 99.40 | 96.20 → 100.00 | 86.18 → 86.41 | warm接纳，三seed等权 |
| 持续欠载 AB1：XY12_32 | 8 / 1 | 99.74 → 99.85 | 100.00 → 100.00 | 未提供同口径实测值 | 全请求CDF，三seed合并 |
| 持续欠载 AB2：XY12_24 | 8 / 1 | 98.28 → 99.13 | 100.00 → 100.00 | 未提供同口径实测值 | 全请求CDF，三seed合并 |
| 持续欠载 AB3：XY12_20 | 8 / 1 | 95.03 → 98.62 | 99.10 → 100.00 | 未提供同口径实测值 | 全请求CDF，三seed合并 |
| 持续欠载 AB4：XY12_16 | 8 / 1 | 90.83 → 97.69 | 93.43 → 99.68 | 未提供同口径实测值 | 全请求CDF，三seed合并 |
| 持续欠载 AB5：X16 | 8 / 1 | 92.56 → 98.33 | 95.89 → 99.73 | 未提供同口径实测值 | 全请求CDF，三seed合并 |
| 持续欠载 E1：200K/20K固定分卡 | 8 / 1 | 83.50 → 96.92 | 22.22 → 100.00 | 未提供同口径实测值 | warm接纳，seed7 |


五组AB的上表SLO与历史CDF一致，使用全部请求；它们的warm SLO在第5节另列。E1全请求SLO×1.5为39.83% → 100%，与表中的warm结果22.22% → 100%不同。

旧AB的Random/Ordered不能归为“全程持续过载”：原逐事件审计显示Random需求过载时间占75.88%，Ordered为34.19%；其理想工作量平均需求124.91 GiB/s略高于120容量，但随时间存在欠载区间。真正以持续过载构造的多画像组是full。

## 2. 输入画像、配比与共同模型

总长度均包含NQL，NQL表示未命中后需要计算的token数量。这里的“随机输入”是**固定配额后逐卡独立随机打乱**：所有请求在t=0进入各自NPU队列，卡内串行接纳。它不是泊松到达，也不代表每次独立同分布抽样。


| 实验 | 画像集合 | 每卡配额/输入配比 | 每卡请求数 | 每seed总数 |
|---|---|---|---|---|
| full | 32/64/80/128/160/200K × NQL256/1024/2048/4096 | 每个长度对应四种NQL为3:2:1:1 | 42 | 1344 |
| semi | 与full相同24画像 | 每个长度对应四种NQL为1:1:1:2 | 30 | 960 |
| near35 | 与full相同24画像 | 每个长度对应四种NQL为1:1:3:2 | 42 | 1344 |
| sensitivity20k_076 | A200K/NQL中心2048；B20K/NQL中心1152，窄幅扰动 | 7A+42B，A:B=1:6，独立洗牌 | 49 | 392 |
| 旧AB Random | A128K/256；B32K/4096 | 40A+80B，A:B=1:2，独立洗牌 | 120 | 3840 |
| 旧AB Ordered | 与旧AB Random相同 | ABB重复40轮 | 120 | 3840 |
| 五组XY Random | 详见第5节 | 每卡A:B=1:12，独立洗牌 | 按组不同 | 按组不同 |
| E1 Fixed | A200K/2048；B20K/1024 | 固定4张卡处理A，4张卡处理B | 不是逐卡混排 | 472 |


full、semi和near35这三组24画像实验沿用条带放置`(block_index+npu_id)%3`，同一block各层同盘，单位为GiB/s；这几组不能写成ring hash。五组XY与E1使用ring hash，256 vnodes，单位为十进制GB/s。虽然两组参数表都写“每盘40、每卡50”，单位不同，不能直接混算容量或读取时间。

这些实验均为8层、batch=1、按层读取与计算重叠。Baseline是path0 FIFO；原始Once按层进行路径选择，使用5ms状态快照。固定候选池＋Once是另一策略，结果单独放在第7节。

四类画像的定义：第一个字母按总长度≤80K为S、>80K为L；第二个字母按NQL<512为S、≥512为L。因此SS/LS是短计算窗口画像，并不表示其KV读取量都小。


| 输入组 | SS请求数占比 | SL请求数占比 | LS请求数占比 | LL请求数占比 |
|---|---|---|---|---|
| full | 21.43% | 28.57% | 21.43% | 28.57% |
| semi | 10.00% | 40.00% | 10.00% | 40.00% |
| near35 | 7.14% | 42.86% | 7.14% | 42.86% |


## 3. 多画像实验的类别级结果

以下均取warm `[2,4)s`，三seed先各自统计再等权平均。类别U是该类在窗口内的计算卡时间/该类活跃卡时间；SLO按该窗口内接纳的请求统计，保留窗后完成的请求。数字单位为%，箭头为Baseline → Once。


| 实验 | 类别 | 类别NPU U | SLO×1.5 |
|---|---|---|---|
| full | SS | 10.91 → 97.86 | 0.00 → 99.19 |
| full | SL | 65.95 → 98.61 | 40.50 → 99.49 |
| full | LS | 22.79 → 78.88 | 0.00 → 84.79 |
| full | LL | 92.65 → 54.53 | 95.08 → 31.11 |
| semi | SS | 65.31 → 99.93 | 65.56 → 100.00 |
| semi | SL | 99.72 → 99.78 | 100.00 → 100.00 |
| semi | LS | 81.90 → 94.82 | 91.53 → 98.15 |
| semi | LL | 99.92 → 99.19 | 100.00 → 100.00 |
| near35 | SS | 61.64 → 99.27 | 61.11 → 100.00 |
| near35 | SL | 99.70 → 99.98 | 100.00 → 100.00 |
| near35 | LS | 79.65 → 98.97 | 80.74 → 100.00 |
| near35 | LL | 99.89 → 99.24 | 100.00 → 100.00 |


full中Once显著改善SS、SL、LS，但LL的SLO×1.5由95.08%降至31.11%，体现了过载时的类别取舍。semi和near35的总体U原本已接近100%，短计算请求得到改善后，按请求数统计的SLO提升明显，而按计算卡时间统计的整机U提升很小。

near35中NQL256请求占总请求数14.29%，仅占纯计算工作量约1.70%，因此不能用“总体U几乎没变”否定其短计算请求改善。

三组总体SLO×1（同一warm口径）为：


| 实验 | SLO×1：Baseline → Once |
|---|---|
| full | 4.92 → 36.65 |
| semi | 69.56 → 79.67 |
| near35 | 76.16 → 84.58 |


## 4. 旧AB：128K/NQL256与32K/NQL4096

该组32 NPU、3 SSU×40 GiB/s，A:B=1:2；Random与Ordered是同一组画像的不同顺序。以下都来自seed7、warm `[2,4)s`。


| 顺序 | 类别 | Baseline U | Once U | Baseline SLO×1.5 | Once SLO×1.5 |
|---|---|---|---|---|---|
| Random | A：128K/256 | 49.13 | 49.97 | 28.70 | 33.96 |
| Random | B：32K/4096 | 99.19 | 99.63 | 100.00 | 100.00 |
| Ordered | A：128K/256 | 15.14 | 32.73 | 0.00 | 25.69 |
| Ordered | B：32K/4096 | 97.10 | 99.78 | 100.00 | 100.00 |


Random总体SLO×1为53.67% → 60.64%。Ordered总体U提升12.41个百分点，SLO×1.5提升3.18个百分点；这些结果不能放到200K/20K组合下。

原SSD物理服务审计给出：Baseline Random实际总供给111.136248 GiB/s，带宽利用率92.613540%；Ordered为72.307936 GiB/s、60.256613%。Once尚未核得同口径SSD汇总。另一个带宽均值表以**NPU实际收到的数据量**积分：Baseline Random总接收111.1463 GiB/s，Ordered为72.2520 GiB/s。HBM接收端与SSD服务端在窗口边界可不同，应保留各自定义；供给与需求整窗均值之比也不等于NPU利用率。

旧AB还存在Baseline三种子均值92.74% / 79.36%，它和上面单seed7的90.68% / 77.42%并不矛盾。原始Once尚无完整三seed对照，不能拿单seedOnce与三seedBaseline作配对结论。

## 5. 从阻塞公式构造的五组持续欠载AB

这五组的正确编号为`XY12_32、XY12_24、XY12_20、XY12_16、X16`。它们都是8 NPU、1 SSU、逐卡A:B=1:12随机混排；不能用包内其它固定并发场景D1、D2或F20替代。

设每层读取量为V、计算时间为C，x=V_A/V_B，y=C_A/C_B。候选组合同时考察需求约束和单A阻塞条件；全程名义欠载不自动意味着“单A一定阻塞B”。第1、2组未满足完整的单A阻塞联合条件，第3、4、5组满足。每组完整轨迹均已逐事件核验普通参考需求不超过盘容量；跨请求L0预取的突发I/O不叠加到该参考需求。


| 组 | ID | A总长度/NQL | B总长度/NQL | x | y | A需求GB/s | B需求GB/s | 实测峰值名义需求GB/s |
|---|---|---|---|---|---|---|---|---|
| 1 | XY12_32 | 200K/3564 | 32K/1455 | 6.43 | 12.00 | 2.3036 | 4.3011 | 32.41 |
| 2 | XY12_24 | 200K/2645 | 24K/1325 | 8.69 | 12.00 | 3.1169 | 4.3013 | 34.41 |
| 3 | XY12_20 | 200K/2189 | 20K/1236 | 10.53 | 12.00 | 3.7735 | 4.3012 | 34.41 |
| 4 | XY12_16 | 200K/1735 | 16K/1122 | 13.31 | 12.00 | 4.7691 | 4.3007 | 37.68 |
| 5 | X16 | 200K/2048 | 16K/1122 | 13.28 | 14.16 | 4.0355 | 4.3007 | 34.14 |


32K组计算画像用了data网格内插值；24K、20K、16K短请求的计算时间由32K与48K线性外推，尚未硬件校准。它们不是全部直接取原data行。读取量按`(总token−NQL)×1408 bytes`计算。

下面明确列出两种SLO口径，防止把历史CDF与warm U写成同一批请求。历史CDF使用三seed合并的全请求，包含冷启动；warm SLO使用`[2,4)s`接纳请求并先逐seed统计再平均。


| ID | warm NPU U | 全请求SLO×1 | 全请求SLO×1.5 | warm SLO×1.5 |
|---|---|---|---|---|
| XY12_32 | 99.74 → 99.85 | 78.53 → 90.62 | 100.00 → 100.00 | 100.00 → 100.00 |
| XY12_24 | 98.28 → 99.13 | 48.91 → 78.40 | 100.00 → 100.00 | 100.00 → 100.00 |
| XY12_20 | 95.03 → 98.62 | 28.08 → 70.64 | 99.10 → 100.00 | 98.57 → 100.00 |
| XY12_16 | 90.83 → 97.69 | 6.62 → 65.44 | 93.43 → 99.68 | 94.87 → 100.00 |
| X16 | 92.56 → 98.33 | 9.56 → 72.92 | 95.89 → 99.73 | 95.13 → 100.00 |


类别级结果如下。类别U仍为warm，SLO列仍为全请求CDF口径；数字单位为%。


| ID | 类别 | warm 类别U | 全请求SLO×1 | 全请求SLO×1.5 |
|---|---|---|---|---|
| XY12_32 | A | 99.93 → 99.75 | 78.12 → 28.12 | 100.00 → 100.00 |
| XY12_32 | B | 99.48 → 100.00 | 78.56 → 95.83 | 100.00 → 100.00 |
| XY12_24 | A | 99.78 → 99.12 | 55.00 → 4.17 | 100.00 → 100.00 |
| XY12_24 | B | 96.43 → 99.14 | 48.40 → 84.58 | 100.00 → 100.00 |
| XY12_20 | A | 99.22 → 98.27 | 4.17 → 4.17 | 100.00 → 100.00 |
| XY12_20 | B | 90.25 → 99.05 | 30.07 → 76.18 | 99.03 → 100.00 |
| XY12_16 | A | 99.23 → 97.34 | 4.86 → 4.86 | 100.00 → 100.00 |
| XY12_16 | B | 83.81 → 98.10 | 6.77 → 70.49 | 92.88 → 99.65 |
| X16 | A | 99.34 → 98.05 | 4.86 → 4.86 | 100.00 → 100.00 |
| X16 | B | 85.19 → 98.68 | 9.95 → 78.59 | 95.54 → 99.71 |


五组CDF的每策略样本数依次为1248、1560、1560、1872、1872；A:B在每组均为1:12。表里的名义需求不能除以40冒充物理盘带宽利用率。原始记录可恢复全程盘利用率，但其窗口含冷启动与排空，与主表warm不同；因此主表不混填，单列如下。

| ID | 全程SSD利用率 Baseline → Once |
|---|---:|
| XY12_32 | 65.47% → 65.55% |
| XY12_24 | 71.90% → 73.17% |
| XY12_20 | 76.26% → 78.96% |
| XY12_16 | 81.17% → 87.82% |
| X16 | 76.18% → 80.83% |

来源是逐次运行的 `ssd_mean_utilization`，再对三seed等权平均；精确值见 [five_ab_results_long.csv](tables/formula_ab/five_ab_results_long.csv)。

## 6. 精确200K/2048＋20K/1024：随机混排筛选与E1固定分卡

这对精确画像另有原始代码和结果；用户指定目标现已确认是第12节sensitivity20k_076，其短请求NQL中心为1152。本节保留精确2048/1024的两类独立实验，集中入口为 [精确AB资料说明](05_exact_ab_200K2048_20K1024/README.md)。此前仅列E1不够完整，新增找回的随机混排如下。

### 6.1 32卡4盘随机混排：含局部过载的一组

原实验 `mixed20_screen`，32 NPU、4 SSU，每盘40 GiB/s、每卡50 GiB/s，8层、seed7。每张卡按固定A:B配额独立打乱整个请求序列，A/B的长度和NQL均无扰动；B的计算耗时由32K/48K数据外推。以下均为**Baseline短窗筛选**，统计窗口 `[0.2,0.8)s`，与本报告其他 `[2,4)s` 结果分开阅读。SLO按窗口内接纳请求统计。

| A:B | 原实验ID | 整机NPU U | A类U | B类U | SLO×1.5 | 任一盘名义需求过载时长 |
|---|---|---:|---:|---:|---:|---:|
| 1:4 | mixed20_s4_L1_S4 | 97.04% | 99.89% | 87.63% | 86.44% | 0 ms |
| 1:8 | mixed20_s4_L1_S8 | 98.69% | 99.69% | 95.46% | 100.00% | 0 ms |
| **1:16** | **mixed20_s4_L1_S16** | **96.37%** | **99.59%** | **93.81%** | **100.00%** | **65.86 ms（窗口的10.98%）** |
| 1:24 | mixed20_s4_L1_S24 | 99.40% | 99.91% | 99.12% | 100.00% | 0 ms |

1:16组前两盘峰值名义需求分别40.51、40.22 GiB/s，超过每盘40 GiB/s容量；四盘平均名义需求分别36.65、35.47、34.19、33.68 GiB/s，因此存在局部过载。该需求定义是当前已接纳请求的每层读取量/自身每层计算时间，L0预取不重复计入；它不是实测供给带宽。

代码、四组冻结输入、原结果和指标见 [mixed20筛选目录](05_exact_ab_200K2048_20K1024/mixed20_screen_exact/README.md)。目前找回的是Baseline，尚未定位这四组的Once对照及匹配原图。已有逐8卡Baseline/Once带宽图属于第10节的1664/1024版本，不能挪作本组图片。

### 6.2 E1：8卡1盘固定分卡欠载

E1为8 NPU、1 SSU，固定4卡持续处理A、4卡持续处理B，seed7；代码、原trace和包含E1的原图见 [E1固定分卡目录](05_exact_ab_200K2048_20K1024/E1_fixed/README.md)。它不是每卡A/B随机混排，也不是前述71.01%的Ordered实验。


| 范围 | 指标 | Baseline | Once |
|---|---|---|---|
| 整机 warm | NPU U | 83.50% | 96.92% |
| A类 warm | NPU U | 100.00% | 100.00% |
| B类 warm | NPU U | 67.00% | 93.85% |
| 整机 warm接纳 | SLO×1.5 | 22.22% | 100.00% |
| A类 warm接纳 | SLO×1.5 | 100.00% | 100.00% |
| B类 warm接纳 | SLO×1.5 | 12.50% | 100.00% |
| 整机 全请求 | SLO×1.5 | 39.83% | 100.00% |
| A类 全请求 | SLO×1.5 | 100.00% | 100.00% |
| B类 全请求 | SLO×1.5 | 34.86% | 100.00% |


E1全请求每策略472条，warm接纳数Baseline144、Once192；最大单盘名义需求37.04 GB/s。原图 `fixed_concurrent_utilization.png` 和 `stall_decomposition.png` 均包含E1及其他组，已原样保留；它们属于E1汇总，不是上述32卡随机筛选的图。其他历史20K随机实验还使用过B的NQL1152，或者A的NQL1664以及扰动，分别保留对应配置。

## 7. 固定候选池＋Once补充对照

它在原始Once之外限制类别可用Path池，应单独命名。已有结果如下，U/SLO均为warm `[2,4)s`：


| 场景 | 种子口径 | NPU U | SLO×1.5 | SSD带宽利用率 |
|---|---|---|---|---|
| full | 三seed等权 | 64.37% | 76.96% | 100.00% |
| semi | 三seed等权 | 99.18% | 100.00% | 83.26% |
| 旧AB Random | seed7 | 85.38% | 91.05% | 未列 |
| 旧AB Random | 三seed等权 | 86.06% | 91.85% | 未列 |
| 旧AB Ordered | 未测试 | — | — | — |


## 8. 统一阅读口径

- **NPU利用率**：窗口内全部NPU计算时间之和，除以NPU数量×窗口长度。类别U的分母是该类别活跃卡时间，所以不能把各类U简单平均得到整机U。
- **TTFT与SLO**：本工程TTFT=请求prefill完成−在NPU上接纳时刻；不包含接纳前的卡内排队。SLO基准=8×单层纯计算时间，CDF横轴`TTFT/SLO`，x=1.5处就是SLO×1.5达标率。
- **窗口样本**：warm SLO选窗口内接纳的请求，并继续跟踪到最终完成。窗口后完成或超时的请求没有被删除。不同策略的warm接纳集合可能不同。
- **三seed聚合**：full、semi、near35的CDF与达标率采用种子等权平均；五组XY历史CDF为三seed全请求直接合并。这两种聚合方法必须保留。
- **带宽**：参考需求是当前已接纳请求的V/C；实际供给来自真实服务区间积分。逐SSD最多40，逐NPU接收最多50，且需看GB/s还是GiB/s。需求超过容量与供给超过容量是两回事。
- **负载分类**：near35在包内替换欠载随机输入条目，其性质是每盘平均需求约35 GiB/s、允许局部过载，不能解释为全程欠载。五组XY核验的是完整轨迹普通名义需求不超过容量；这也不意味着没有跨请求预取突发或排队。sensitivity20k_076按本次整理要求列为局部过载补充AB，其普通需求欠载和预取突发约束单列于第12节。
- **启动/排空**：完整批次与warm窗口包含的计算、等待和请求集合不同。不得把完整批次U与warm SLO拼成同一统计窗口的结果。

## 9. 图片、数据和代码入口

链接指向已经找回或从既有结果补生成的文件。原图与新增图分开列出；`NEW_`前缀表示本次按冻结输入补画，并非重新仿真。


| 材料 | 文件 |
|---|---|
| full 两策略SLO CDF | [full_random_ttft_ratio_cdf_two_strategies.png](01_continuous_overload/full24/cdf/full_random_ttft_ratio_cdf_two_strategies.png) |
| full Baseline逐盘带宽 | [full_baseline_per_ssu.png](repository_source/results/diverse_data_ssu3_l3_20260916/figures/full_baseline_per_ssu.png) |
| semi 两策略SLO CDF | [semi_random_ttft_ratio_cdf_two_strategies.png](02_partial_overload/semi24/cdf/semi_random_ttft_ratio_cdf_two_strategies.png) |
| semi 全部32卡计算时序 | [semi_2-4s_全部NPU计算时序_卡秒.png](02_partial_overload/semi24/figures/semi_2-4s_全部NPU计算时序_卡秒.png) |
| semi 250ms带宽与计算面积图 | [semi_250ms_带宽与NPU利用率_面积标注.png](02_partial_overload/semi24/figures/semi_250ms_带宽与NPU利用率_面积标注.png) |
| semi 逐卡等待原因 | [semi_2-4s_逐NPU等待原因.md](02_partial_overload/semi24/figures/semi_2-4s_逐NPU等待原因.md) |
| near35 SLO CDF | [near35_ttft_slo_cdf.png](03_continuous_underload/near35/near35_random_results/figures/near35_ttft_slo_cdf.png) |
| near35 整机需求与供给 | [near35_fleet_seed7.png](03_continuous_underload/near35/near35_random_results/figures/near35_fleet_seed7.png) |
| near35 Baseline逐盘带宽 | [near35_baseline_per_ssu_seed7.png](03_continuous_underload/near35/near35_random_results/figures/near35_baseline_per_ssu_seed7.png) |
| near35 Once逐盘带宽 | [near35_once_per_ssu_seed7.png](03_continuous_underload/near35/near35_random_results/figures/near35_once_per_ssu_seed7.png) |


旧AB的图：


| 材料 | 文件 |
|---|---|
| Random两策略SLO×1/×1.5 CDF | [AB_random_TTFT_CDF_SLO_1_1p5.png](02_partial_overload/ab128_32/cdf/AB_random_TTFT_CDF_SLO_1_1p5.png) |
| Baseline Random计算时序 | [baseline_random.png](repository_source/results/baseline_ab128_32_ratio12_20260912/figures/ssu3/baseline_random.png) |
| Baseline Ordered计算时序 | [baseline_ordered.png](repository_source/results/baseline_ab128_32_ratio12_20260912/figures/ssu3/baseline_ordered.png) |
| Once Random计算时序 | [once_random.png](repository_source/results/baseline_ab128_32_ratio12_20260912/once_per_layer_ssu3_seed7/figures/once_random.png) |
| Once Ordered计算时序 | [once_ordered.png](repository_source/results/baseline_ab128_32_ratio12_20260912/once_per_layer_ssu3_seed7/figures/once_ordered.png) |
| Baseline Random逐卡层带宽 | [random_all_32npu_layer_average.png](repository_source/results/baseline_ab128_32_ratio12_20260912/figures/ssu3/random_all_32npu_layer_average.png) |
| Baseline Ordered逐卡层带宽 | [ordered_all_32npu_layer_average.png](repository_source/results/baseline_ab128_32_ratio12_20260912/figures/ssu3/ordered_all_32npu_layer_average.png) |
| Once Random逐卡层带宽 | [random_all_32npu_layer_average.png](repository_source/results/baseline_ab128_32_ratio12_20260912/once_per_layer_ssu3_seed7/figures/random_all_32npu_layer_average.png) |
| Once Ordered逐卡层带宽 | [ordered_all_32npu_layer_average.png](repository_source/results/baseline_ab128_32_ratio12_20260912/once_per_layer_ssu3_seed7/figures/ordered_all_32npu_layer_average.png) |


五组AB最终版SLO横轴CDF（图例“流量分配”，每图全部请求）：


| 组 | 文件 |
|---|---|
| XY12_32 | [ttft_slo_all_01_XY12_32.png](03_continuous_underload/formula_ab/ttft_slo_all_requests_images/ttft_slo_all_01_XY12_32.png) |
| XY12_24 | [ttft_slo_all_02_XY12_24.png](03_continuous_underload/formula_ab/ttft_slo_all_requests_images/ttft_slo_all_02_XY12_24.png) |
| XY12_20 | [ttft_slo_all_03_XY12_20.png](03_continuous_underload/formula_ab/ttft_slo_all_requests_images/ttft_slo_all_03_XY12_20.png) |
| XY12_16 | [ttft_slo_all_04_XY12_16.png](03_continuous_underload/formula_ab/ttft_slo_all_requests_images/ttft_slo_all_04_XY12_16.png) |
| X16 | [ttft_slo_all_05_X16.png](03_continuous_underload/formula_ab/ttft_slo_all_requests_images/ttft_slo_all_05_X16.png) |


本次从冻结输入补画的随机混排图：


| 材料 | 文件 |
|---|---|
| full详细画像混排 | [NEW_full_seed7_profile_shuffle.png](tables/derived/other_scenarios/NEW_full_seed7_profile_shuffle.png) |
| semi详细画像混排 | [NEW_semi_seed7_profile_shuffle.png](tables/derived/other_scenarios/NEW_semi_seed7_profile_shuffle.png) |
| near35详细画像混排 | [NEW_near35_seed7_profile_shuffle.png](tables/derived/other_scenarios/NEW_near35_seed7_profile_shuffle.png) |


复核与复现入口：


| 用途 | 文件 |
|---|---|
| full / semi 总与类别原表 | [partial_threeway_macro_summary.csv](repository_source/results/diverse_data_ssu3_l3_20260916/threeway_macro_summary.csv) |
| full / semi CDF样本SLO复核 | [overload_recomputed_slo.csv](tables/overload_recomputed_slo.csv) |
| near35报告 | [near35_report.md](03_continuous_underload/near35/near35_random_results/near35_report.md) |
| near35源码/结果绘图入口 | [report_near35.py](03_continuous_underload/near35/report_near35.py) |
| near35带宽补列表 | [near35_summary_with_bandwidth.csv](03_continuous_underload/near35/near35_summary_with_bandwidth.csv) |
| 五组XY完整数据包说明 | [README.md](03_continuous_underload/formula_ab/original_recovered/README.md) |
| 五组XY完整结果长表 | [five_ab_results_long.csv](tables/formula_ab/five_ab_results_long.csv) |
| 精确200K/2048＋20K/1024全部已找回资料 | [README.md](05_exact_ab_200K2048_20K1024/README.md) |
| 精确AB随机混排短窗筛选 | [README.md](05_exact_ab_200K2048_20K1024/mixed20_screen_exact/README.md) |
| E1固定分卡代码、trace和原图 | [README.md](05_exact_ab_200K2048_20K1024/E1_fixed/README.md) |
| E1固定分卡精确结果 | [E1_fixed_reference.csv](tables/formula_ab/E1_fixed_reference.csv) |
| 旧AB精确指标 | [ttft_slo.csv](repository_source/results/baseline_ab128_32_ratio12_20260912/once_per_layer_ssu3_seed7/ttft_slo.csv) |
| 旧AB运行入口 | [experiment.py](repository_source/results/baseline_ab128_32_ratio12_20260912/experiment.py) |


near35包包含3种子×2策略完整result与manifest；六个result的SHA256均与原包记录吻合。五组XY包还含固定并发和改变y的其它探索组，它们作为原始材料保留，但不混入本报告的五组结果。

精确200K/2048＋20K/1024已确认有四组32卡随机混排Baseline筛选，以及8卡固定分卡E1双策略结果；“71.01% / 71.43%”仍明确属于旧128K/32K的Ordered实验。尚未找到精确200K/2048＋20K/1024随机混排的Once配对原图。

旧AB四份大体积`result.json.gz`本次未能完整离线取回，不能视作已随包包含；已保留输入manifest、命令、汇总表、原图、61份校验后的源码/资料，以及原结果的精确来源路径。其代码和图表可以查阅，离线完整重绘依赖补齐这些原始result。


## 10. 找回的逐8卡带宽图：mixed8原始版本

已找回的 `fifo_8npu_bandwidth_simple.png` / `once_8npu_bandwidth_simple.png` 与原图包逐字节一致；实际属于 **A≈200K/NQL1664、B≈20K/NQL1024，窄幅扰动，每卡6A+66B**。A的NQL范围1651–1664，B为1024–1106；不是精确A200K/2048+B20K/1024的E1，也不是NQL中心2048/1152的另一敏感性组。

本组mixed8是另一实验，不是用户确认的sensitivity20k_076。为防止再混用，实际输入范围保留如下。范围来自冻结输入的`metadata.json/profiles`，不是由图片刻度估读；单位为**GiB/s**。

| 类别 | 实际NQL范围 | 每层读取量 | 每层计算时间 | 需求范围 GiB/s | 换算 GB/s |
|---|---:|---:|---:|---:|---:|
| A / 长请求 | 1651–1664 | 0.266372680664 GiB | 57.0524–57.5039 ms | 4.6323–4.6689 | 4.9738–5.0132 |
| B / 短请求 | 1024–1106 | 0.0255126953125 GiB | 5.2440–5.6668 ms | 4.5022–4.8652 | 4.8342–5.2239 |

上述范围仅用于辨认mixed8这组原图。原图横轴2–4秒、橙线预取窗口需求（含L0）、蓝线2ms实际接收带宽。常规V/C任意8卡组合上界38.9165 GiB/s，小于盘容量40；包含下一请求L0的预取窗口仍可产生局部拥塞，应与32卡逐盘名义需求局部过载区分。

本组8 NPU、1 SSU×40 GiB/s、seed7、warm[2,4)s，结果如下：

| 指标 | FIFO | Once |
|---|---:|---:|
| 整机NPU利用率 | 92.5240% | 97.1557% |
| SLO×1.5 | 98.2143% | 100.0000% |
| 短请求类别利用率 | 85.2652% | 97.5435% |
| 长请求类别利用率 | 99.2896% | 96.8690% |
| 真实盘带宽利用率 | 86.6321% | 92.1784% |

图、2ms带宽CSV、576条原输入与请求执行数据、CDF点、原代码，均保存在 [逐8卡原图与输入映射说明](04_ab8_bandwidth_original/mapping.md) 所列目录。两个策略各自的逐8卡需求/供给和整机供给图可在 `index.html` 的“八卡带宽原图”组直接浏览。

简图橙线按下一层读取量/当前计算窗口计需求，含L0预取；蓝线按2ms真实字节计算。盘端服务与NPU接收有传输时差，应保留两者定义。这与多画像图的“当前已接纳请求V/C”参考需求也不同。

## 11. 原始仓库与资料完整性

- [原始多画像实验目录](https://github.com/chguo0503/qos_storage_sim/tree/1c2bb30acdf8c721476b0294cb31079f0fab6fba/results/diverse_data_ssu3_l3_20260916)
- [旧AB指标及Ordered归属的原始说明](https://github.com/chguo0503/qos_storage_sim/blob/1c2bb30acdf8c721476b0294cb31079f0fab6fba/results/baseline_ab128_32_ratio12_20260912/ssu3_three_strategy_metrics.md)
- [全部仓库路径、版本与是否已打包](repository_files_index.csv)
- [文件完整性与复现说明](FILE_COMPLETENESS.md)
- [全部归档文件及SHA256](file_inventory.csv)

五组AB的原ZIP缺少中央目录，本次从局部条目恢复469个文件，全部通过CRC/长度检查；72份原始压缩trace与原验证记录SHA256一致。再从其它原始资料补回56份SHA256完全相同的代码，主实验入口 `run_experiment.py` 的依赖导入已通过，未重跑仿真。完整报表批处理尚有缺失文件，详见完整性说明。


## 12. 局部过载补充AB：2026-09-14 sensitivity20k_076

用户提供实验ID后，已按原始配置、原生数据和图片审计记录确认目标为 `lower_fifo_followup_20260914/native20/sensitivity20k_076_seed7_{fifo,once}`。按本次整理要求，该组加入局部过载目录。其普通名义需求保持欠载，跨请求预取存在突发；这里的目录归类不改变原实验的需求定义和约束。

[完整说明与代码入口](02_partial_overload/sensitivity20k_076/README.md) · [三张原图浏览页](02_partial_overload/sensitivity20k_076/index.html) · [指标CSV](tables/sensitivity20k_verified_metrics.csv)

配置：8 NPU、1 SSU×40 GiB/s、每卡接收上限50 GiB/s、8层、seed7，每卡7个A+42个B，独立随机混排，共392请求。A总长度200K、NQL中心2048（实际2042–2054）；B总长度20K、NQL中心1152（实际1126–1178）。`076`为候选枚举编号，不是0.76系数。B计算时间由32K/48K数据外推，原图保留敏感性实验标记。

| 2–4s指标 | Baseline/FIFO | Once |
|---|---:|---:|
| 整机NPU利用率 | 94.6147% | 99.2567% |
| A类NPU利用率 | 99.4464% | 98.9465% |
| B类NPU利用率 | 86.7022% | 99.8375% |
| 全部warm接纳请求TTFT SLO×1.5 | 98.4375%（126/128） | 100%（137/137） |
| A类warm SLO×1.5 | 100%（18/18） | 100%（18/18） |
| B类warm SLO×1.5 | 98.1818%（108/110） | 100%（119/119） |
| SSU实际平均带宽 | 30.0216 GiB/s | 31.3784 GiB/s |
| SSU实际带宽利用率 | 75.0540% | 78.4460% |

原CDF使用相同全部392请求，SLO×1.5为98.7245%→100%，与warm口径不同；SLO×1、各类计数均已保存在指标CSV。

中心画像单请求需求A=3.758370 GiB/s、B=4.309402 GiB/s，实际需求范围见 [画像表](tables/sensitivity20k_verified_profiles.csv)。全程当前请求名义需求峰值34.1539 GiB/s，低于40；跨长短请求交界预取仍有突发，总预取需求峰值117.1386→114.6358 GiB/s。原实验采用“普通需求全程欠载，交界预取豁免”的统计约束；真实SSD供给按2ms字节计量，两策略峰值均不超过40 GiB/s。

已找回并逐字节验证的原图：

| 内容 | 原图 |
|---|---|
| 两策略整机需求与真实SSD供给 | [sensitivity20k_bandwidth_pair.png](02_partial_overload/sensitivity20k_076/original_figures/sensitivity20k_bandwidth_pair.png) |
| 两策略全部8卡计算与IO等待时序 | [sensitivity20k_timeline_pair.png](02_partial_overload/sensitivity20k_076/original_figures/sensitivity20k_timeline_pair.png) |
| 整体/A/B的TTFT CDF | [sensitivity20k_ttft_cdf_pair.png](02_partial_overload/sensitivity20k_076/original_figures/sensitivity20k_ttft_cdf_pair.png) |

原代码包和图片包均完整，CRC及图片原SHA256核对通过。本次未重跑、未重画。现存原包中尚未找到可验证属于该实验的独立逐NPU需求/供给PNG；原数据保留逐NPU的2ms实际接收字节和逐层时间，未用mixed8逐卡图替代。
