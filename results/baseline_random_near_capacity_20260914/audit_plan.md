# Baseline Random 近容量研究：独立审查与执行建议

更新时间：2026-09-13T15:59:36.315012+00:00

本审查核对源码、输入、工作量和运行环境。只研究 Baseline Random；这里提到 Once 是解释既有策略差异，并不授权新增 Once 或 Ordered 仿真。核心源码没有修改。

## 1. Baseline 实际是什么

`shared_path_baseline.baseline_path_ids()` 把所有 IO 放到各自磁盘的 Path 0。`sim.PathQueue.enqueue()` 追加到 deque，`activate_next()` 从左边取出，因此每盘独立 FCFS/FIFO，**并非所有盘共用一个全局 FIFO**。

`sim.DiskIOScheduler._dispatch_one()` 每盘同时只执行一条命令；非抢占地以 40 GiB/s 读取。每张卡还有独立 50 GiB/s 的 FCFS 接收链路。请求层读取完整到齐后才能计算下一层；计算当前层时启动下一层读取，当前请求末层计算可以预取下个请求首层。

同 Path 的大读取可能先发出许多 176 KiB 命令，后来短计算请求的读取排在这些命令后面；这才是存储队头阻塞。不是一张卡的长计算占用了另一张卡的计算器。

剩余服务机会在活跃 Path 间分配；当前 FINAL_STATIC 的 PIR 全是无穷，CIR 是保证权重，不是不可借用上限。不存在已验证的“A/B 固定限速，富余带宽无法转移”的实现。

来源：`sim.py` 的 `PathQueue`、`_static_qos_service_rates`、`_dispatch_one`；`shared_path_baseline.py`；`continuous_prefill_client.py`；`continuous_batch_sim.py`。

## 2. 新 runner 与旧 runner 的输入约束

新文件 `experiment.py` 按每卡 `Random(seed + 100003*npu)` 打乱完整请求池，保持每卡整数画像比例，符合独立随机要求。旧根文件 `run_baseline_npu32_stress.build_workload()` 的非 fixed 模式只打乱一个短 deck 再循环，不能直接把它叫作本次完整池随机。

调用根 `run_case()` 时必须显式给 `strategy='baseline'`、`assignment='fixed'`、`windows=((2000,4000),(2000,20000))`。默认策略是 baseline_native，默认窗口是 [1,2)/[2,3) 秒，两者都不能误用。

coflow 路径实际固定使用：8 层、batch=1、每盘40 GiB/s、每卡50 GiB/s、5ms状态采集、0.1us逐命令发出、启用跨请求首层预取。仅改 metadata 不会改变这些固定参数。输入需要精确176KiB块，且每层placement体积与读取V相符。

新 runner 用未经缩放的 data 行；每请求总长至少10K，miss数量与总长分别记录。每卡纯计算配额至少22秒可保证不会在20秒前因请求耗尽而闲置，但不能保证 warm 两秒内实际看到所有画像。后者必须从实际计算层区间逐卡检查，不能用全队列画像计数代替。

warm 利用率应始终是：

```text
U_window = 窗口内32张卡实际计算毫秒数之和 / (32 * 2000ms)
```

不能删掉 IO stall、空闲卡或只统计完整落入窗口的层。现有 summarize_window() 按边界裁剪计算，满足这一要求。

## 3. 工作守恒为什么限制解释空间

对同一批完整请求，纯计算总量 W_C 和各盘读取量 W_V[d] 已由输入确定。

```text
T_end >= max_d(W_V[d] / 40GiB/s)
T_end >= max_i(该卡纯计算总时间)
整批 U = W_C / (32*T_end)
```

旧 A=(128K,miss256)、B=(32K,miss4096)，每卡A:B=1:2时：

- 理想无等待平均需求为124.9097586 GiB/s，3盘容量为120 GiB/s，负载比1.0409147。
- 3840个请求的 W_C=647267.2823331915 NPU*ms、读取2526.5625 GiB。
- 最热盘读取给出 T_end>=21057.373046875ms，故**整个有限批次** U<=96.0571033%。
- 旧 Random seed7 的整批 U=89.0497940%，理论容量上限仍比它高约7.007个百分点；所以“盘接近满就证明 Baseline 最优”也不成立。
- 旧三个随机种子的 [2,20) 秒 U 平均91.4144113%；相同长期混合比例下容量已限制整体提升空间，但不能直接把有限批次96.057%界硬套到任意两秒 warm 窗口。窗口里画像占比和预取库存都可能不同。

当前搜索更换画像或比例，会同时更改 W_C/W_V，所以必须重新算每个候选的负载比和上界。不能拿旧上界约束新输入，也不能把总过载造成的低 U 全归因于 FIFO。

来源：旧研究 `comparison.json`、`formula_review/accounting_checks.json` 的3盘Random seed7记录；已核对其manifest与源码记录。

## 4. 画像变更会改变 Once 的路径池

分类逻辑是总输入<=80K为第一位S，否则L；miss>=512为第二位L，否则S。它不是按V/C直接分类。

FINAL_STATIC 共有8组，每组SS/SL/LS/LL路径数为12/4/12/4；对应全盘类别保证量20/6/8/6 GiB/s。所有路径PIR无穷。

旧A=(128,256)属于LS、B=(32,4096)属于SL；新main512的大请求(192,4096)属于LL，短计算请求(32,512)属于SL。将短计算画像改成(32,256)还会从SL变成SS。Baseline始终只用Path0，分类本身不改变它的路径选择；但未来若和Once比较，类别池和CIR配置变化是额外变量，不能声称仅改变流大小。

## 5. 冻结与远端对照

已核对当前29个核心/策略源文件和9月12日运行记录逐字节一致，根 stress runner SHA256也是原值94704edf908fbf90b9e39f5b26f94ad59c0c5dcfc4c1b981e1a2860921c2946d。

`source_files()` 的29文件不含根 `run_baseline_npu32_stress.py` 与 `run_multi_ssu_stall_experiments.py`，不能只靠29哈希覆盖新runner全部依赖。根代理将在独立 `study_plan.json` 冻结整个根目录.py+data及新runner，不修改正在运行的脚本；每个case结束应与plan核对。

远端只使用新建独立/tmp目录，复制根目录.py、data和本研究runner；没有递归复制旧results、没有修改他人进程。已存在输入应直接复制冻结manifest，禁止远端用不同Python重新生成随机排列后冒充同一输入。

远端环境详情与所有传输文件hash记录在 `audit_remote_setup.json`。凭据不写入任何文件。

## 6. 本地与远端资源、加速建议

2026-09-14只读快照：

| 环境 | CPU | 内存总量 | 可用内存 | swap |
|---|---:|---:|---:|---:|
| 本机 | 20逻辑CPU | 约22GiB | 约12GiB | 2GiB中已用1.9GiB |
| 192.168.31.126 | 20逻辑CPU | 45.74GiB | 42.28GiB | 8GiB全空 |

远端初始loadavg约0.05，系统Python3.14.4，没有numpy。已获授权安装隔离的3.10环境及numpy2.2.6。uv没有3.10.10可用下载，使用可用3.10补丁版本，先运行同一冻结manifest验证跨机一致性再分配矩阵。

建议本地先用2～3进程；本次远端已确定最多6进程，并按此上限冻结队列。内存和桌面响应比CPU核数更可能先成为限制。大trace渲染不要与许多仿真进程同时运行。

安全优先的加速：

1. 不留逐块trace的筛选，只保留固定窗口实际服务积分；最终候选再补trace。
2. 每个case独立进程并行，使用冻结manifest；numpy/BLAS线程设为1以免外部线程超额竞争。
3. 保持现有精确块、发出间隔、5ms采集和完整运行语义。把176KiB合并成大块、移除Path仲裁、切baseline_native或停在warm结束，都不是已经证明等价的加速。
4. 可以另行基准测试GC阈值/关闭周期GC等仅影响Python执行的措施，但先验证输出一致、监控峰值内存；本审查没有采用这些未验证优化。
5. 若仅为先筛选warm表现，可以减少纯计算配额至覆盖warm结束的安全余量；例如旧ABB配额9轮纯计算4551.098ms。但重新shuffle更小池会改变输入，不能视作原40轮输入的同一随机前缀；本轮已采用22秒配额，不建议运行中调整。

当前新runner物理积分按完成回调回溯SSD/link真实服务区间，完整排空后统计，因此跨窗口晚完成的IO不会漏记。它仍有每块回调成本；不应为了快把缺失流量当0或只统计窗口内完成的块。

## 7. 已冻结候选的容量算术（seed7）

以下仅由输入计算；不是仿真结果，不证明瞬时欠载。整批U上界采用最热盘工作量和最长卡纯计算量中的较强下界。

| 候选 | 理想平均需求/总容量 | 最热盘理想平均 GiB/s | 整批U容量上界 |
|---|---:|---:|---:|
| reference | 104.091% | 41.641897 | 96.0571% |
| main512 | 99.731% | 39.892845 | 100.0000% |
| main1024 | 100.197% | 40.086557 | 99.7841% |
| tight256 | 97.966% | 39.191983 | 100.0000% |
| load095 | 95.149% | 38.060041 | 100.0000% |
| load104 | 104.117% | 41.647201 | 96.0449% |
| smaller_blocker | 99.895% | 39.958998 | 100.0000% |
| three_profile | 100.040% | 40.021302 | 99.9468% |

运行时补充：远端已安装Python3.10.19和numpy2.2.6，main512 seed7完整跨机验证通过。所有保存的科学字段、各请求/层指标、各窗口、固定时间段物理服务积分均逐值完全相等，只有显式主机/路径/Python版本与真实执行耗时元数据不同。无trace运行未保留逐块事件时戳，因此不声称比较了没有保存的逐块事件。完整证据见 `runtime_validation/main512_cross_host_checks.json`。单个匹配案例支持使用此远端环境，不等于已经证明任何未来输入都必然一致。

远端随后按 `audit_remote_queue_plan.json` 完成21个不同的Baseline Random案例，最多6并行，不保留trace。源文件、每个输入和队列顺序已冻结；没有远端重新随机生成输入。`audit_remote_queue_status.json`记录运行状态，`audit_remote_sync_record.json`记录每个已完成案例的导入时间及hash。导入前复核输入、核心/runner源码hash、结果hash、物理块守恒及全部模拟器不变量，输出不会覆盖本地已有案例。已用 `audit_remote_acceptance.py`再次独立读取全部21份最终产物，`audit_remote_acceptance.json`显示全部通过。

## 8. 另行授权的两例Once探索配对

根据Baseline pilot结果另外选择reference seed7、adapt_f2_01 seed7，各运行一次Once per layer，不保留trace。它们使用各自Baseline的原始冻结manifest字节，结果保存在对应 `runs/<label>/once/`，不计入Baseline随机种子数量。这里是看到pilot后做的探索配对，不能包装成事先指定的无偏策略总体评估。

独立的 `audit_remote_once_plan.json`、`audit_remote_once_queue.py`、`audit_remote_once_status.json`、`audit_remote_once_sync_record.json` 和 `audit_remote_once_acceptance.json`记录此批工作。原21案例计划保持不变。新队列只有在原队列无pending任务后才利用空位启动，每次启动记录原Baseline占位数；两种策略远端合计最多6个仿真进程，Once自身最多2个。

配对验收再次核对源hash、原始输入字节、请求fingerprint、全部物理块计数和模拟器不变量。冻结observer中的 `baseline_nonzero_paths`只对Baseline累加；Once产物里它为0不表示Once用了Path0，也不会把这一字段当作Once路径正确性的证明。

## 9. 透明构造输入的远端四例

另外授权的四个Baseline Random案例为raw200K/miss4096长请求，配合12K或16K/miss128短请求，分别用6或8个SSU。短请求计算时间由固定miss128数据的仿射拟合向32K以下外推，必须标为构造输入，不能称为data的实测短请求。条带余数可能使部分盘理想平均需求略超40GiB/s，整机理想平均接近容量也不等于逐盘逐时刻欠载。

远端使用独立 `audit_remote_constructed_plan.json`，其中冻结114个核心/构造依赖文件和四个最终输入压缩包hash。独立 `constructed_input_audit.json`逐请求验证通过后才生成 `audit_remote_constructed_launch.json` 并启动。它核对最终短请求 `source_ttft_ms=None`，不把外推纯计算时间包装成原data TTFT。

这批队列同时读取前面两个队列的占位，两者均无pending任务后才利用空位，所有远端仿真合计最多6个。四例已完成并通过 `audit_remote_constructed_acceptance.json` 验收。

## 10. 最后四例长读取绝对规模对照

独立 `audit_remote_context_scale_plan.json` 固定8盘、S10K/miss128，长请求为200/256/384/512K、固定miss1024。200K长请求保留原data行，256/384/512K长计算时间和全部10K短计算时间均明确为外推。121个源/构造/审计依赖与四份输入文件已冻结，`context_input_audit.json`独立逐请求审计通过后才启动。

仅512K案例按事先授权保留1.8–4.2s物理trace；其余三例没有trace。`audit_remote_trace_check.py`将在远端逐条检查保留行的请求/层/块身份、条带映射、去重、窗口相交、块大小、SSD/link物理服务时长和来源hash；本地再核对传回trace的SHA256，避免为校验而在本地重复解码大文件。保留行是窗口子集，完整运行的观察块总数另行核对，不能把两种数量混淆。

案例完成后先由 `audit_remote_context_preview.py`独立裁剪原始计算区间，报告warm/long利用率及每卡长短实际计算覆盖，再执行完整trace验收。用户后来将目标放宽为长期80–89.99%；仍保留原输入与固定seed7结果，不据此改输入或替换随机种子，确认矩阵由根代理另行冻结。

所有队列结束后，`audit_remote_concurrency.py`会按31个任务的实际启动/确认结束时间统一复核远端同时运行不超过6个仿真。
