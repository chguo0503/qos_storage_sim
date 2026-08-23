# sim.py 离散事件 QoS 仿真详解（入门版）

## 1. 这份程序在模拟什么

这份程序模拟的是：许多请求需要从多块磁盘读取分层 KV 数据，然后交给 NPU 逐层计算。多个请求同时运行时，它们会竞争磁盘队列和带宽，因此会产生 I/O 等待、NPU 空闲和请求排队。

可以把系统想象成一家工厂：

- 请求是订单。
- NPU 是加工机器。
- 每一层 KV 数据是加工前必须取到的原料。
- 磁盘是原料仓库。
- KV 块读取流是一次次取料任务。
- QoS 队列是仓库里的不同服务窗口。
- 全局调度器决定订单交给哪台机器。
- 事件堆是按时间排序的日程表。

程序已经包含完整的事件循环入口 <code>simulate_continuous()</code>。它会一直处理事件，直到所有请求完成，并返回 NPU 状态与统计结果。

## 2. 它是“离散事件”仿真，不是固定时间步仿真

虽然日常也会把它笼统叫作离散时间仿真，但代码实际上采用的是离散事件仿真。

固定时间步仿真可能每隔 0.01 ms 检查一次所有对象。离散事件仿真只跳到下一个真正发生变化的时刻。

例如：

- 当前时间为 2.00 ms；
- A 流会在 2.05 ms 完成；
- B 流会在 2.30 ms 完成。

程序会直接从 2.00 ms 跳到 2.05 ms，处理 A 的完成事件，不会遍历 2.01、2.02、2.03、2.04 ms。

这样做的优点是：没有状态变化的时间段不需要计算，特别适合大量 I/O 流和分层计算任务。

## 3. 最重要的单位与公式

| 物理量 | 代码中的单位 | 典型成员 |
| --- | --- | --- |
| 时间 | ms | <code>current_time</code>、<code>end_time</code>、<code>ttft_ms</code> |
| 原始每层计算时间 | µs | <code>per_layer_us</code> |
| 数据量 | GB | <code>per_layer_kv_gb</code>、<code>remaining_gb</code> |
| 带宽 | GB/s | <code>DISK_BW</code>、<code>bw</code>、<code>required_bw</code> |
| 序列长度 | K token | <code>seq_len_k</code> |

I/O 时间的基本公式是：

    I/O 时间(ms) = 数据量(GB) / 带宽(GB/s) × 1000

例如，读取 0.002 GB，队列带宽为 0.5 GB/s：

    0.002 / 0.5 × 1000 = 4 ms

每层计算时间进入事件循环前会从微秒换算成毫秒：

    compute_ms = per_layer_us / 1000

## 4. 整体结构

    bw_table
        │
        ▼
    generate_npu_loads() ──► 请求负载字典
        │
        ▼
    GlobalScheduler ──► 选择实例和空闲 NPU
        │
        ▼
    build_block_placement() ──► 每层 KV 块及其磁盘位置
        │
        ▼
    start_kv_load() ──► BlockIOFlow
        │
        ▼
    DiskQueue ──► DiskIOScheduler ──► DiskState
        │                                  │
        └──────── DISK_COMPLETION 事件 ────┘
                                           │
                                           ▼
                                      NPUState
                                           │
                                  COMPUTE_DONE 事件
                                           │
                                           ▼
                                      下一层或完成

理解代码时，先记住下面五个职责：

| 对象 | 它主要回答的问题 |
| --- | --- |
| <code>NPUState</code> | 这个 NPU 的当前请求执行到哪一层了？ |
| <code>BlockIOFlow</code> | 这个 KV 块还剩多少数据没有读完？ |
| <code>DiskQueue</code> | 这个逻辑 QoS 队列里谁正在服务、谁在等待？ |
| <code>DiskState</code> | 这块物理磁盘当前有哪些活动流？ |
| <code>GlobalScheduler</code> | 新请求应该交给哪个实例和哪个 NPU？ |

## 5. 输入数据：bw_table 和请求负载

### 5.1 bw_table

带宽表的键是 <code>(seq_len_k, nql)</code>，值是四元组：

    bw_table[(seq_len_k, nql)] = (
        required_bw,
        per_layer_us,
        ttft_ideal_ms,
        per_layer_kv_gb,
    )

四个值分别表示：

| 字段 | 含义 |
| --- | --- |
| <code>required_bw</code> | 请求的带宽需求，用于选择 QoS 层级 |
| <code>per_layer_us</code> | NPU 计算 nql 个 token 时，一层计算需要多少微秒 |
| <code>ttft_ideal_ms</code> | 理想 TTFT，主要作为参考指标 |
| <code>per_layer_kv_gb</code> | 其余 token 每层需要从 SSU 读取多少 GB 的 KV 数据 |

两个键的准确含义是：

- <code>seq_len_k</code> 表示这次请求的总输入长度，单位为 K token；
- <code>nql</code> 表示总输入中需要在 NPU 侧计算的 token 数，单位为“个”。

因此一条请求会被拆成：

    total_input_tokens = seq_len_k × 1024
    npu_compute_tokens = nql
    ssu_read_tokens = seq_len_k × 1024 - nql

例如 <code>seq_len_k=100, nql=256</code> 表示总输入为 102400 个 token，其中 256 个由 NPU 计算，其余 102144 个需要从 SSU 读取。

数据表中的 <code>per_layer_kv_gb</code> 已经是这部分 <code>ssu_read_tokens</code> 对应的总数据量，不应再乘一次比例。全部 84 组数据的 <code>per_layer_kv_gb / ssu_read_tokens</code> 都等于约 1311.302 bytes/token。

因为采用 layerwise 推理，NPU 计算当前层的时间就是 SSU 读取下一层数据的目标窗口：

    required_bw = per_layer_kv_gb / (per_layer_us × 10⁻⁶)

表中的 <code>per_layer_us</code> 已经是 NPU 计算 nql 个 token 时一层所需的总时间，仿真不会再次按 nql 缩放。

<code>load_bw_table_cache()</code> 优先读取 NPZ 缓存；缓存不存在时会读取项目根目录的 <code>data</code>。仿真核心只要求调用者最终传入一个符合上面结构的字典。

### 5.2 generate_npu_loads()

这个函数从带宽表中选择请求，生成如下负载字典：

    {
        'npu_id': 0,
        'seq_len_k': 32,
        'nql': 64,
        'required_bw': 20.0,
        'per_layer_us': 1000.0,
        'ttft_ideal_ms': 10.0,
        'per_layer_kv_gb': 0.002,
        'category': 'SS',
    }

主要参数：

| 参数 | 作用 |
| --- | --- |
| <code>load_profile='mixed'</code> | 从完整表中随机选择 |
| <code>load_profile='heavy'</code> | 偏向较重负载 |
| <code>load_profile='light'</code> | 偏向较轻负载 |
| <code>ls_ratio</code> | 控制混合负载中不同类别的比例 |
| <code>total_bw_cap</code> | 必要时删除高需求请求，使需求总和不超过上限 |
| <code>num_npu</code> | 这里表示要生成多少条负载；在完整仿真中实际传入总请求数 |

完整仿真还会为每条请求补上：

| 字段 | 含义 |
| --- | --- |
| <code>request_id</code> | 请求唯一编号 |
| <code>arrival_time</code> | 请求到达系统的时间 |
| <code>instance_id</code> | 调度后所属实例 |
| <code>npu_id</code> | 调度后实际使用的 NPU |

### 5.3 请求分类

<code>classify_request()</code> 按总输入长度和 NPU 计算 token 数把请求分成四类：

| 条件 | 类别 |
| --- | --- |
| 短输入、较少 NPU 计算 token | SS |
| 短输入、较多 NPU 计算 token | SL |
| 长输入、较少 NPU 计算 token | LS |
| 长输入、较多 NPU 计算 token | LL |

默认边界是：

- <code>seq_len_k &lt;= 80</code> 为短序列；
- <code>nql &gt;= 512</code> 为高 nql。

SS、SL、LS、LL 是负载分类。它们和后面按 <code>required_bw</code> 划分的 QoS 带宽层级不是同一件事。

## 6. KV 块放置：build_block_placement()

只有需要从 SSU 读取的 token 会生成 KV 块：

    n_ssu_tokens = seq_len_k × 1024 - nql
    n_blocks = ceil(n_ssu_tokens / 128)

每个完整块包含 128 个 token。最后不足 128 token 时，最后一块按真实 token 数计算数据量，而不是强行补成完整块。

例如 <code>seq_len_k=1, nql=300</code>：

    总输入 = 1024 token
    NPU 计算 = 300 token
    SSU 读取 = 724 token
    块数 = ceil(724 / 128) = 6

前 5 块各包含 128 token，最后一块包含 84 token。所有块的数据量之和始终等于表中的 <code>per_layer_kv_gb</code>。

返回结构是：

    placement[npu_id][layer] = [
        {'disk': disk_id, 'gb': block_size_gb},
        ...
    ]

支持四种放置模式：

| 模式 | 规则 |
| --- | --- |
| <code>random</code> | 每块随机选择磁盘 |
| <code>roundrobin</code> | 按块编号轮流选择磁盘 |
| <code>local_balanced</code> | 优先选择当前请求内部累计数据最少的磁盘 |
| <code>load_aware</code> | 优先选择全局累计数据最少的磁盘 |

<code>simulate_continuous()</code> 在请求真正分配到某个 NPU 时才为它建立放置结果，因此放置中的 NPU 编号与实际调度结果一致。

## 7. 事件与事件堆

每个事件是一个五元组：

    (
        event_time,
        event_type,
        resource_id,
        value,
        generation,
    )

Python 的最小堆先比较时间。时间相同时，再比较事件类型等后续字段。

完整事件循环实际处理六类事件：

| 事件 | 资源编号 | value | 作用 |
| --- | --- | --- | --- |
| <code>REQUEST_ARRIVAL</code> | request_id | 0 | 请求进入全局等待队列 |
| <code>DISK_COMPLETION</code> | disk_id | 0 | 一块磁盘上一个或多个流完成 |
| <code>BATCH_DISPATCH</code> | npu_id | layer | 按固定时间间隔提交该层的下一批块 |
| <code>TOKEN_REFILL</code> | disk_id | 0 | 更新可选的磁盘令牌桶 |
| <code>COMPUTE_DONE</code> | npu_id | layer | 某 NPU 完成一层计算 |
| <code>DISK_REBALANCE</code> | disk_id | 0 | 合并处理同一时刻对同一磁盘的带宽重分配请求 |

<code>KV_BLOCK_DONE</code>、<code>NPU_START</code> 和 <code>NPU_RESTART</code> 仍作为兼容常量保留，但当前事件循环不创建它们。

### 7.1 generation 为什么重要

某个流原来预计在 5 ms 完成。如果 2 ms 时磁盘带宽发生变化，它的完成时间可能变成 8 ms。原来的 5 ms 事件已经过期，但它仍留在堆中。

每次磁盘重新分配带宽时都会增加 <code>generation</code>：

1. 新事件携带新版本号；
2. 旧事件携带旧版本号；
3. 事件弹出时比较版本号；
4. 版本不一致就把它计为 <code>stale_events</code> 并忽略。

这避免了频繁从堆中搜索并删除旧事件。

## 8. BlockIOFlow：一个 KV 块读取任务

<code>BlockIOFlow</code> 表示“读取某一请求、某一层中的一个块”，或者一组能够等价合并的虚拟块。只有同一 NPU、同一层、同一 SSU、相同块大小且在同一时刻提交的块才会合并。

### 8.1 主要成员

| 成员 | 含义 |
| --- | --- |
| <code>flow_id</code> | 全局递增的流编号 |
| <code>npu_id</code> | 这个块属于哪个 NPU 当前处理的请求 |
| <code>layer</code> | 所属层 |
| <code>block_idx</code> | 层内块编号 |
| <code>disk_id</code> | 从哪块磁盘读取 |
| <code>total_gb</code> | 一个块或一组虚拟块合计的原始总大小 |
| <code>remaining_gb</code> | 尚未完成的数据量 |
| <code>bw</code> | 当前实际带宽 |
| <code>start_time</code> | 当前这段带宽开始生效的时间 |
| <code>end_time</code> | 按当前带宽估算的完成时间 |
| <code>demand_bw</code> | 请求的需求带宽，用于选择 QoS 层级 |
| <code>queue_id</code> | 所属逻辑队列 |
| <code>active</code> | 流是否仍然有效 |
| <code>block_count</code> | 这条对象代表的虚拟块数量；普通流为 1 |
| <code>_queue</code> | 所属 <code>DiskQueue</code> 的反向引用 |

### 8.2 update_bw()

当带宽变化时，不能直接用新带宽覆盖旧带宽。必须先结算旧带宽已经传输了多少数据：

    done_gb = old_bw × (current_time - start_time) / 1000

然后：

1. 从 <code>remaining_gb</code> 中减去已完成数据；
2. 如果数据已经读完，将流标记为完成；
3. 如果新带宽为 0，将流暂停，完成时间设为无穷大；
4. 否则按新带宽重新计算完成时间。

完整的磁盘调度器通过 <code>settle()</code> 批量执行同样的结算思想。

## 9. DiskState：一块物理磁盘

<code>DiskState</code> 保存整块磁盘的动态状态，不负责决定 QoS 规则。

| 成员 | 含义 |
| --- | --- |
| <code>disk_id</code> | 磁盘编号 |
| <code>disk_bw</code> | 物理磁盘带宽 |
| <code>active_flows</code> | 当前真正接受服务的流 |
| <code>generation</code> | 当前磁盘事件版本 |
| <code>busy_time</code> | 有活动流的累计时间 |
| <code>idle_time</code> | 没有活动流的累计时间 |
| <code>last_event_time</code> | 磁盘统计结算到的时间 |
| <code>surplus_bw_integral</code> | 未使用带宽乘时间的累计值 |
| <code>total_bw_integral</code> | 总物理带宽乘时间的累计值 |
| <code>queue_scheduler</code> | 该磁盘对应的 <code>DiskIOScheduler</code> |

<code>add_flow()</code> 和 <code>remove_flow()</code> 维护活动流以及忙闲统计。<code>DiskIOScheduler</code> 在每次状态变化前调用 <code>settle()</code>，所以同一段时间不会被重复累计。

## 10. DiskQueue：一条逻辑 QoS 队列

<code>DiskQueue</code> 是单块磁盘上的一条逻辑队列。

### 10.1 主要成员

| 成员 | 含义 |
| --- | --- |
| <code>queue_id</code> | 队列编号 |
| <code>group_id</code> | 队列所属的带宽组编号 |
| <code>cir</code> | 队列活跃时保证的默认带宽 |
| <code>pir</code> | 峰值带宽；QoS 队列当前为无穷大，即 uncapped |
| <code>wrr_weight</code> | 借用剩余磁盘带宽时使用的 WRR 权重 |
| <code>pending</code> | 尚未激活的 FCFS 等待队列 |
| <code>active_flows</code> | 当前允许接受服务的流 |
| <code>assigned_bw</code> | 当前给整条队列的带宽 |
| <code>max_depth</code> | 队列最多同时激活多少条流 |
| <code>bytes_served</code> | 已完成的总数据量 |
| <code>n_activations</code> | 流被激活的次数 |

### 10.2 为什么所有硬件队列都使用 max_depth=4

QoS 模式下，每条队列使用 FCFS：

1. 新流进入 <code>pending</code> 尾部；
2. 只要当前活动数不足 4，就继续从队头激活；
3. 已经有 4 条活动流时，后续流留在 <code>pending</code>；
4. 一条活动流完成后，再按 FCFS 从队头补一条。

因此一条队列任意时刻最多有 4 条活动流。queue_wrr 中，带宽先分给整条队列：队列获得 CIR，并可按 WRR 权重借用剩余带宽；随后这条队列的带宽再按虚拟块数量分给它的 1 到 4 条活动流。<code>pending</code> 中的流已经提交到队列，但还不传输，也不参与当前带宽分配。

fair 基线虽然只有一条共享队列，但它仍代表同一种硬件队列，因此也使用 <code>max_depth=4</code>：最多 4 条流共同公平分享整块 SSU 带宽，其余流留在同一条 path 的 FCFS pending 区。

## 11. QoS 的 CIR 保底与 PIR-uncapped 借用规则

这是当前实现最重要的规则。

### 11.1 24 队列：三档带宽组

24 队列配置中，每块磁盘有 3 个带宽层级，每层 8 条队列：

| required_bw 条件 | 层级 | 队列编号 | 每条队列 CIR | WRR 权重 |
| --- | ---: | --- | ---: | ---: |
| <code>&gt; 100 GB/s</code> | 0 | 0 到 7 | 2.0 GB/s | 4 |
| <code>&gt; 10 GB/s</code> 且 <code>&lt;= 100 GB/s</code> | 1 | 8 到 15 | 0.5 GB/s | 2 |
| <code>&lt;= 10 GB/s</code> | 2 | 16 到 23 | 0.1 GB/s | 1 |

层内队列偏移量为：

    (npu_id + layer) % 8

例如，一个中档请求由 NPU 0 处理：

- 第 0 层进入队列 8；
- 第 1 层进入队列 9；
- 第 2 层进入队列 10。

这些队列是共享的，不是每个 NPU/DPU 一条专属 path。例如 NPU 0 的 L0 和 NPU 8 的 L0 都进入队列 8；它们按 FCFS 先后接受服务。同一个 NPU 也会随着层号变化在本层级的 8 条队列之间轮换。

### 11.2 24 队列的空闲带宽如何借用

分配分为两步：

1. 每条活跃队列先获得自己的 CIR；
2. 剩余物理磁盘带宽按照活跃队列的 WRR 权重分配。

QoS 队列的 PIR 是无穷大，因此不会把借用带宽限制在 <code>required_bw</code>。只要队列里还有数据，单个活跃队列就可以一直借到物理磁盘上限。例如只有中档队列 8 活跃时，它会从 0.5 GB/s 的 CIR 增长到完整的 40 GB/s。

如果一条高档、一条中档、一条低档队列同时活动，它们分别得到：

    CIR 合计 = 2.0 + 0.5 + 0.1 = 2.6 GB/s
    剩余带宽 = 40 - 2.6 = 37.4 GB/s
    剩余带宽按 4:2:1 分配

最终大约为：

- 高档队列：23.371 GB/s；
- 中档队列：11.186 GB/s；
- 低档队列：5.443 GB/s。

三者合计仍严格等于 40 GB/s。

24 条默认队列带宽之和为：

    8 × 2.0 + 8 × 0.5 + 8 × 0.1 = 20.8 GB/s

20.8 GB/s 是所有队列同时活跃时需要保证的 CIR 总和，不是队列能使用的总上限。默认物理磁盘是 40 GB/s，因此还可借用 19.2 GB/s。如果传入的 <code>disk_bw</code> 小于 20.8 GB/s，程序会直接报错，因为这时无法兑现全部 CIR 保证。

### 11.3 256 队列：8 个等权 group

256 队列配置不沿用上面的高、中、低三档，而是把每块 SSU 的队列均分为 8 个 group：

| 项目 | 数值 |
| --- | ---: |
| group 数量 | 8 |
| 每个 group 的队列数 | 32 |
| 每个 group 的 WRR 权重 | 1 |
| 每个 group 的 CIR 总额 | 5 GB/s |
| 每条队列的 CIR | 0.15625 GB/s |
| 每条队列的 PIR | 无穷大（uncapped） |

每块磁盘的 256 条队列恰好共同保证 40 GB/s：

    256 × 0.15625 = 40 GB/s

这里的 40 GB/s 仍是 CIR 总额，不是把未活跃队列的额度锁死。实际分配时：

1. 每条活跃队列先获得 0.15625 GB/s CIR；
2. 剩余带宽在有活跃队列的 group 之间按 1:1:1……分配；
3. 一个 group 得到的剩余带宽，再由该 group 内的活跃队列均分；
4. 某个 group 或队列空闲时，它没有使用的带宽可被其他活跃 group 和队列借走。

当前代码把 256 条队列作为所有 NPU/DPU 共享的 path，并采用下面的均衡映射：

    group_id = npu_id % 8
    group 内队列 = (npu_id // 8 + layer) % 32
    queue_id = group_id × 32 + group 内队列

这样 128 个 NPU 均匀落入 8 个 group，同一个 NPU 的不同层会在其 group 内轮换队列。用户已经明确 8×32 和权重 1；由于现有代码里没有硬件提供的 DPU 到队列映射表，上述共享映射是本次实验采用的明确假设。如果真实硬件另有映射，只需替换 <code>queue_id_for_flow()</code> 中这一小段规则。

### 11.4 queue_wrr 名称与当前语义

<code>queue_wrr</code> 会先满足 CIR，再使用 <code>wrr_weight</code> 分配所有可借用的剩余带宽。PIR uncapped 表示队列没有单独的峰值上限，但所有活跃队列的总带宽仍不能超过 <code>disk_bw</code>。

当前 <code>urgency_driven</code> 也使用相同的 CIR + WRR + uncapped PIR 规则。两者暂时只保留策略名称上的区别。

### 11.5 fair 基线

<code>fair</code> 不使用上述 24 或 256 条 QoS 队列。每个 SSU 恰好只有一条 QoS path，也就是队列 0：

1. 把该 SSU 上的所有流都放进队列 0；
2. 按 FCFS 最多激活 4 条流，其余流等待；
3. 4 条以内的活动流共同竞争；
4. 以整块磁盘的 <code>disk_bw</code> 为容量；
5. 使用最大最小公平方式给活动流分带宽；
6. 不超过各流的 <code>demand_bw</code>。

为了加速，若干相同大小的块拥有完全相同的生命周期时，程序会用一条聚合流表示它们。<code>block_count</code> 同时作为最大最小公平的权重。例如，一条代表 3 个块的聚合流与一条代表 1 个块的流竞争 40 GB/s 时，二者分别得到 30 GB/s 和 10 GB/s。这等价于原来的 4 条独立流各得 10 GB/s，不会把三块错误地当成一个竞争者。

<code>demand_driven</code> 当前作为同一公平基线的别名。

## 12. DiskIOScheduler：磁盘 I/O 调度器

每块磁盘都有一个 <code>DiskIOScheduler</code>，它连接 <code>DiskQueue</code> 和 <code>DiskState</code>。

主要方法：

| 方法 | 作用 |
| --- | --- |
| <code>enqueue_many()</code> | 把一组新流放入对应队列 |
| <code>settle()</code> | 按旧带宽把所有活动流推进到当前时刻 |
| <code>complete_ready_flows()</code> | 取出已经完成的流并激活队列下一条流 |
| <code>request_redistribution()</code> | 请求在当前时刻重新分配带宽，并合并同一 SSU 的重复请求 |
| <code>redistribute()</code> | 按当前策略重新设置每条活动流的带宽 |
| <code>_schedule_next_event()</code> | 安排最近磁盘完成事件和可选令牌事件 |

<code>redistribute()</code> 的顺序非常重要：

1. 先结算旧带宽产生的进度；
2. 处理令牌桶补充；
3. 计算新带宽；
4. 重算每条流的 <code>end_time</code>；
5. 增加磁盘版本号；
6. 把新的最早完成事件放入堆。

一次块完成可能让许多状态同时变化。程序不会为同一 SSU、同一时间点的每次变化立即重复执行上述过程，而是只创建一个 <code>DISK_REBALANCE</code> 事件。这样仍在相同仿真时刻完成重分配，但显著减少重复扫描与过期完成事件。

## 13. IOSchedulingConfig、批次与令牌桶

### 13.1 IOSchedulingConfig

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| <code>io_dispatch_mode</code> | <code>all_at_once</code> | 一层中的块怎样提交 |
| <code>batch_size</code> | 8 | 每批提交多少块 |
| <code>batch_interval_mode</code> | <code>fixed</code> | 使用固定间隔还是由 NPU 按需求计算间隔 |
| <code>batch_dispatch_interval_us</code> | 200.0 | 相邻两批的固定发送间隔 |
| <code>batch_dispatch_headroom</code> | 1.0 | 自适应模式的目标带宽倍率 |
| <code>batch_min_dispatch_interval_us</code> | 0.0 | 自适应间隔下限 |
| <code>batch_max_dispatch_interval_us</code> | 无穷大 | 自适应间隔上限 |
| <code>qos_queue_max_active_flows</code> | 4 | 每条硬件队列最多同时服务多少条流，fair 也适用 |
| <code>token_bucket_enabled</code> | False | 是否开启磁盘令牌桶 |
| <code>token_bucket_refill_us</code> | 80.0 | 令牌补充周期 |
| <code>token_bucket_pir_cap</code> | True | 令牌耗尽时是否暂停传输 |
| <code>qos_queue_count</code> | 24 | 每块 SSU 配置的 QoS 队列数 |
| <code>qos_layout</code> | <code>three_tier</code> | 队列布局；可选三档或八组布局 |

### 13.2 三种块派发模式

| 模式 | 行为 |
| --- | --- |
| <code>all_at_once</code> | 一次创建并提交该层所有块流 |
| <code>batched</code> | 第 1 批立即提交，之后按原块顺序和所选间隔模式提交 |
| <code>traffic_aware_batched</code> | 派发时优先选择当前压力较小的磁盘上的块，间隔模式与普通 batch 相同 |

这里的 Batch(8) 是整个 NPU 的一层每次总共发送 8 个块，每块对应 128 token；不是每个 SSU、每条 queue 各发送 8 个块。一层可以有很多块，它们按这个全层批次大小逐步发送。

流量感知压力是“活动虚拟块数量 + 队列等待虚拟块数量”。它是轻量级启发式算法，不是未来负载预测。

<code>BatchState</code> 保存当前层派发到第几批、尚未提交哪些块。traffic-aware 模式按 SSU 保存待派发块的双端队列，再用最小堆比较各 SSU 的压力；不需要为了选每个块反复扫描和删除整个剩余列表。调度器还直接维护每块 SSU 的 outstanding 虚拟块计数，因此选批时不必反复扫描所有队列。

发送下一批只等待所选的发送间隔，不等待上一批传完。所以当一次传输超过这个间隔时，多批会自然重叠，已提交块会让 active 数量、pending 数量和队列压力逐步增加。每条新 flow 的 <code>demand_bw</code> 仍复制请求在 <code>data</code> 表中的固定值；批次重叠不会自动把后续批次升级到更高 QoS 档位。

<code>pending_blocks[layer]</code> 始终表示整层尚未完成的块数，其中既包含已提交块，也包含未来批次尚未提交的块。只有所有批次都已发送且全部块都传完，该层才会变成 KV ready。

### 13.3 固定间隔与 NPU 自适应间隔

<code>batch_interval_mode='fixed'</code> 保留原来的固定节奏。当前对照实验使用 200 µs：

    下一批时间 = 当前批时间 + 200 µs

<code>batch_interval_mode='demand_aware'</code> 让每个 NPU 根据刚刚发出的这一批实际有多少 GB，以及本请求在 <code>data</code> 表中的 <code>required_bw</code>，独立计算下一批间隔：

    target_bw = required_bw × batch_dispatch_headroom
    interval_us = batch_gb / target_bw × 1,000,000
    interval_us = clamp(interval_us, min_interval_us, max_interval_us)

例如一批 8 个 128-token 块共有 0.016 GB，请求需求为 20 GB/s，headroom 为 1.1，则下一批约在：

    0.016 / (20 × 1.1) × 1,000,000 = 727.27 µs

高带宽需求请求会选择更短间隔，低带宽需求请求会选择更长间隔。<code>main.py</code> 的自适应实验把 headroom 设为 1.1，表示比表中需求快 10% 发出，用来留出调度和竞争余量；1.1 是可调实验参数，不是硬件常量。最小和最大间隔默认不额外裁剪，未来有真实硬件 doorbell 或提交速率限制时可以直接设置这两个边界。

这里调节的是“什么时候创建下一批 flow”，不会修改 flow 的 <code>demand_bw</code>，也不会根据排队时长更换 QoS 档位。普通 batch 与 traffic-aware batch 都支持这两种间隔模式。

### 13.4 TokenBucket

令牌桶是每块磁盘一个，不是每条 QoS 队列一个。

- 每个补充周期拥有 <code>disk_bw × interval</code> 的数据预算；
- capped 模式下，令牌耗尽后所有流暂停到下一次补充；
- uncapped 模式只记录令牌，不限制传输；
- 默认关闭。

24 队列配置的 CIR 总和是 20.8 GB/s，256 队列配置的 CIR 总和是 40 GB/s；两者都允许活跃队列借用空闲带宽，并且总量不会超过默认磁盘的 40 GB/s。令牌桶默认关闭；开启 capped 令牌桶后，它可能成为额外限制。这里的磁盘令牌桶 capped/uncapped 与 QoS 队列的 PIR uncapped 是两个不同概念。

## 14. NPUState：一个 NPU 的完整状态

<code>NPUState</code> 同时保存当前请求状态和跨请求累计统计。

### 14.1 当前请求身份与输入

| 成员 | 含义 |
| --- | --- |
| <code>npu_id</code> | NPU 编号 |
| <code>instance_id</code> | 所属实例 |
| <code>current_load</code> | 当前请求负载字典 |
| <code>seq_len_k</code> | 总输入长度，单位为 K token |
| <code>nql</code> | 需要在 NPU 侧计算的 token 数 |
| <code>total_input_tokens</code> | <code>seq_len_k × 1024</code> |
| <code>npu_compute_tokens</code> | 等于 <code>nql</code> |
| <code>ssu_read_tokens</code> | 总 token 数减去 NPU 计算 token 数 |
| <code>category</code> | SS、SL、LS 或 LL 负载类别 |
| <code>required_bw</code> | 请求需求带宽 |
| <code>per_layer_us</code> | 每层计算时间 |
| <code>per_layer_kv_gb</code> | 每层 KV 数据量 |
| <code>block_placement</code> | 当前请求的块放置 |

### 14.2 层推进状态

| 成员 | 初始值 | 含义 |
| --- | ---: | --- |
| <code>kv_loaded_up_to</code> | -1 | 已连续完成 KV 加载的最高层 |
| <code>compute_done_up_to</code> | -1 | 已完成计算的最高层 |
| <code>pending_blocks</code> | 空字典 | 每层尚未完成的块数 |
| <code>active_block_flows</code> | 空字典 | 每层已经派发且未完成的流 |
| <code>_kv_ready_layers</code> | 空集合 | 所有块已经完成的层 |
| <code>_compute_active</code> | False | 当前是否有一层正在计算 |
| <code>started</code> | False | 当前请求是否已开始 |
| <code>done</code> | False | 当前请求是否完成 |

<code>kv_loaded_up_to</code> 强调“连续”。例如第 2 层先完成、第 1 层还没完成时，第 2 层会存在于 <code>_kv_ready_layers</code>，但 <code>kv_loaded_up_to</code> 不会跳过第 1 层。

### 14.3 时间和 TTFT

| 成员 | 含义 |
| --- | --- |
| <code>arrival_time</code> | 请求到达系统的时间 |
| <code>request_start_time</code> | 请求拿到 NPU 并开始加载的时间 |
| <code>queueing_delay_ms</code> | <code>request_start_time - arrival_time</code> |
| <code>processing_ttft_ms</code> | 从拿到 NPU 到完成的时间 |
| <code>ttft_ms</code> | 从请求到达到完成的总时间，包含排队 |
| <code>compute_end_time</code> | 该 NPU 最近一次计算完成的全局时间 |
| <code>last_compute_end_time</code> | 上一层计算结束时间 |

关系是：

    ttft_ms = queueing_delay_ms + processing_ttft_ms

### 14.4 I/O 与计算统计

| 成员 | 含义 |
| --- | --- |
| <code>io_wait_L0_ms</code> | 第 0 层等待 KV 的时间 |
| <code>io_wait_L1_ms</code> | 第 1 层等待 KV 的时间 |
| <code>io_wait_L2plus_ms</code> | 第 2 层及以后等待 KV 的总时间 |
| <code>current_request_io_waits</code> | 当前请求逐层等待时间 |
| <code>current_request_kv_actual_dur</code> | 当前请求逐层 KV 加载时长 |
| <code>total_compute_ms</code> | 该 NPU 跨所有请求累计的计算时间 |
| <code>npu_idle_ms</code> | 该 NPU 跨请求累计的 I/O 等待时间 |
| <code>layer_trace</code> | 开启 trace 时记录每层 KV 和计算起止时间 |

### 14.5 历史统计

| 成员 | 含义 |
| --- | --- |
| <code>request_count</code> | 这个 NPU 一共处理了多少请求 |
| <code>ttft_list</code> | 每条已完成请求的总 TTFT |
| <code>per_request_io_detail</code> | 每条请求的完整统计字典 |
| <code>_request_archived</code> | 防止同一请求被重复归档 |
| <code>_batches_dispatched</code> | 跨请求累计派发批次数 |

### 14.6 reset_for_next_request()

一个请求完成后，NPU 不会被销毁，而是通过这个方法复用：

1. 必要时归档旧请求；
2. 使旧流失效；
3. 替换请求参数和块放置；
4. 把层进度重置为 -1；
5. 清空当前请求的块、等待、批次和追踪信息；
6. 保留跨请求累计统计；
7. 设置新请求开始时刻。

## 15. GlobalScheduler：请求级调度

<code>GlobalScheduler</code> 负责“请求交给谁”，不负责“磁盘流得到多少带宽”。

### 15.1 两级结构

- L1：选择实例；
- L2：在实例中选择一个空闲 NPU。

实例配置示例：

    instance_config = {
        0: [0, 1, 2, 3],
        1: [4, 5, 6, 7],
    }

每个 NPU 编号必须出现且只能出现一次。

### 15.2 L1 策略

| 策略 | 含义 |
| --- | --- |
| <code>round_robin</code> | 在有空闲 NPU 的实例间轮询 |
| <code>least_loaded</code> | 选择当前活动 NPU 较少的实例 |
| <code>length_grouped</code> | 按请求长短倾向不同实例 |
| <code>pressure_balanced</code> | 按活动请求需求带宽估算压力 |

### 15.3 L2 策略

| 策略 | 含义 |
| --- | --- |
| <code>round_robin</code> | 在空闲 NPU 中轮询 |
| <code>least_loaded</code> | 选择历史未完成计数最小者 |
| <code>random</code> | 随机选择空闲 NPU |

### 15.4 全局等待队列

所有 NPU 都忙时，新请求进入 <code>_global_pending_queue</code>。该队列是 FIFO：

1. 请求完成后释放 NPU；
2. 立即查看全局队头；
3. 把能调度的请求依次派给空闲 NPU；
4. 为每条新请求重新创建块放置并启动第 0 层 I/O。

<code>num_requests_per_npu</code> 只用来计算总请求数。请求不是提前固定给某个 NPU，而是由运行时调度器动态分配。

## 16. start_kv_load()：启动一层 KV 读取

这个函数把一层的放置描述转换成真正的 I/O 流：

1. 检查层编号，并防止同一层重复启动；
2. 读取该层所有块；
3. 初始化 <code>pending_blocks[layer]</code>；
4. 根据派发模式选第一批块；
5. 为每个块创建 <code>BlockIOFlow</code>；
6. 用 <code>flow_to_queue_id()</code> 选择队列；
7. 按磁盘分组提交；
8. 对分批模式按固定或 NPU 自适应间隔安排下一个 <code>BATCH_DISPATCH</code>；
9. 让受影响磁盘重新计算带宽与下一个事件。

如果一层没有块，它会直接被标为 KV ready。

## 17. 预取模式与顺序模式

### 17.1 prefetch

请求开始时只读取 L0。L0 完成并开始计算后，才同时读取 L1：

    请求开始 → 加载 L0 → 计算 L0
                              └─ 同时加载 L1

以后每开始计算第 k 层，就启动第 k+1 层的 SSU 读取。<code>per_layer_us</code> 是这次读取可以与计算重叠的时间窗口。

如果下一层读取在当前层计算结束前完成，NPU 可以无缝进入下一层；否则 NPU 从当前层计算结束开始等待，直到下一层 I/O 完成。

### 17.2 sequential

请求开始时只加载 L0。每层计算完成后，才启动下一层 KV：

    加载 L0 → 计算 L0 → 加载 L1 → 计算 L1 → ...

顺序模式更容易理解，但 NPU 通常会等待更多 I/O。

## 18. 完整事件循环：simulate_continuous()

### 18.1 初始化

函数会：

1. 校验参数；
2. 生成总请求负载；
3. 创建 NPU、实例调度器和磁盘；
4. 为每块磁盘创建 I/O 调度器；
5. 把所有请求到达事件放入最小堆；
6. 从时间 0 开始弹出事件。

默认第 i 条请求的到达时间为：

    i × arrival_interval_ms

默认间隔为 0，所以所有请求同时到达。

### 18.2 REQUEST_ARRIVAL

1. 请求进入全局 FIFO；
2. 调度器寻找空闲 NPU；
3. 记录排队时间；
4. 构建块放置；
5. 只启动 L0；L1 会在 L0 开始计算时启动。

### 18.3 DISK_COMPLETION

1. 检查磁盘版本；
2. 把所有流推进到事件时刻；
3. 找出同一时刻完成的全部流；
4. 从磁盘和队列移除它们；
5. 在 active 数量不足 4 时，从同队列的 FCFS 等待区补充流；
6. 减少 NPU 对应层的 <code>pending_blocks</code>；
7. 整层完成时标记 KV ready；
8. 如果下一计算层已经 ready，启动计算；
9. 为磁盘重新安排完成事件。

### 18.4 BATCH_DISPATCH

1. 用 NPU 编号、层号和批次序号检查事件是否仍有效；
2. 按普通顺序或 traffic-aware 规则选择下一批；
3. 立即把这一批提交到对应 SSU 的队列；
4. 如果还有未发送块，按固定或 NPU 自适应间隔安排下一个批次事件。

这个事件与 <code>DISK_COMPLETION</code> 相互独立，因此前一批仍在传输时也能发送后一批。

### 18.5 COMPUTE_DONE

1. 检查 NPU 计算版本；
2. 标记当前层计算完成；
3. 顺序模式此时才启动下一层 I/O；prefetch 模式已在本层计算开始时启动；
4. 如果还有层，尝试启动下一层计算；
5. 最后一层完成时计算 TTFT 并归档；
6. 释放 NPU；
7. 立即调度全局等待队列中的下一请求。

### 18.6 TOKEN_REFILL

令牌桶开启时：

1. 推进流到当前时间；
2. 补充令牌；
3. 恢复或调整流带宽；
4. 重新安排完成与令牌事件。

### 18.7 终止条件

循环在 <code>completed_requests == total_requests</code> 时结束。

两种保护会主动报错：

- 还有请求没完成，但事件堆已经空了：说明仿真死锁；
- 处理事件数达到 <code>max_events</code>：防止逻辑错误导致无限循环。

## 19. 一个可以手算的完整例子

假设：

- 1 个 NPU；
- 1 块 40 GB/s 磁盘；
- 2 层；
- 总输入为 1K token，其中 300 个由 NPU 计算、724 个从 SSU 读取；
- 请求 <code>required_bw = 20 GB/s</code>；
- 每层需要从 SSU 读取 0.02 GB；
- 每层计算为 1 ms；
- 使用 QoS CIR + WRR + PIR-uncapped 和 prefetch。

第一步，请求属于中档层级，所以队列 CIR 为 0.5 GB/s、WRR 权重为 2。

第二步，NPU 0 的：

- L0 进入队列 8；
- L1 进入队列 9。

第三步，t=0 时只读取 L0。此时整块磁盘只有队列 8 活跃，因此它从 0.5 GB/s CIR 借到完整的 40 GB/s：

    每层 I/O = 0.02 / 40 × 1000 = 0.5 ms

表中需求带宽也正好满足：

    0.02 GB / 0.001 s = 20 GB/s

第四步：

| 时间 | 发生的事 |
| --- | --- |
| 0 到 0.5 ms | 读取 L0 |
| 0.5 ms | L0 ready，开始计算 L0，同时开始读取 L1 |
| 0.5 到 1.0 ms | 计算 L0；读取 L1 |
| 1.0 ms | L1 已经 ready，等待 L0 计算结束 |
| 1.0 到 1.5 ms | 继续计算 L0 |
| 1.5 到 2.5 ms | 计算 L1 |
| 2.5 ms | 请求完成 |

因此：

    processing_ttft_ms = 2.5 ms
    queueing_delay_ms = 0 ms
    ttft_ms = 2.5 ms

如果改成 sequential：

    L0 I/O 0.5 + L0 计算 1 + L1 I/O 0.5 + L1 计算 1 = 3 ms

prefetch 把完整的 L1 I/O 与 L0 计算重叠，将总时间从 3 ms 降为 2.5 ms。项目测试用例会自动验证这两个结果。

## 20. simulate_continuous() 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| <code>bw_table</code> | 必填 | 请求性能表 |
| <code>policy</code> | <code>queue_wrr</code> | QoS 或公平策略 |
| <code>num_requests_per_npu</code> | 1 | 总请求数乘数 |
| <code>num_npu</code> | 32 | NPU 数量 |
| <code>num_disk</code> | 8 | 磁盘/SSU 数量 |
| <code>n_layers</code> | 8 | 仿真层数 |
| <code>ls_ratio</code> | None | 混合负载比例 |
| <code>rng</code> | 固定种子随机源 | 负载和放置随机数 |
| <code>io_sched</code> | 默认配置 | 批次和令牌配置 |
| <code>placement_mode</code> | <code>random</code> | 块放置策略 |
| <code>disk_bw</code> | 40.0 | 每块磁盘带宽 |
| <code>io_mode</code> | <code>prefetch</code> | 预取或顺序 I/O |
| <code>load_profile</code> | <code>mixed</code> | 负载类型 |
| <code>total_bw_cap</code> | None | 请求需求总和限制 |
| <code>instance_config</code> | 单实例包含全部 NPU | 实例到 NPU 的映射 |
| <code>l1_strategy</code> | <code>round_robin</code> | 实例选择策略 |
| <code>l2_strategy</code> | <code>round_robin</code> | NPU 选择策略 |
| <code>arrival_interval_ms</code> | 0.0 | 相邻请求到达间隔 |
| <code>max_events</code> | 10,000,000 | 事件数保护上限 |
| <code>trace</code> | False | 是否保存逐层详细时间 |

总请求数通常是：

    num_npu × num_requests_per_npu

如果设置了 <code>total_bw_cap</code>，负载生成器可能删除一部分请求，因此实际总数可能更少。

## 21. 返回值和 summary

调用返回：

    npus, summary = simulate_continuous(...)

<code>npus</code> 是所有 <code>NPUState</code> 对象，适合查看每个 NPU 的历史与利用率。

<code>summary</code> 的顶层主要字段：

| 字段 | 含义 |
| --- | --- |
| <code>policy</code> | 策略名 |
| <code>io_dispatch_mode</code> | 一次性、普通分批或 traffic-aware 分批 |
| <code>batch_interval_mode</code> | 固定或按请求需求自适应的间隔模式 |
| <code>batch_dispatch_interval_us</code> | 分批模式的固定发送间隔 |
| <code>batch_dispatch_headroom</code> | 自适应目标带宽相对 <code>required_bw</code> 的倍率 |
| <code>batch_min_dispatch_interval_us</code> | 自适应间隔下限 |
| <code>batch_max_dispatch_interval_us</code> | 自适应间隔上限 |
| <code>qos_queue_count</code> | 配置的 QoS 队列数 |
| <code>qos_queue_max_active_flows</code> | 每条硬件队列的活动流上限，fair 也适用 |
| <code>qos_layout</code> | 三档或八组布局 |
| <code>qos_group_queue_counts</code> | 每个 group 的队列数 |
| <code>qos_group_cir_gbps</code> | 每个 group 中单条队列的 CIR |
| <code>qos_group_weights</code> | 每个 group 的借用权重 |
| <code>fixed_qos_queue_bandwidth</code> | 兼容字段；PIR-uncapped 实现中固定为 False |
| <code>qos_cir_guaranteed</code> | 当前策略是否为每条 QoS 队列保证 CIR |
| <code>qos_surplus_borrowing</code> | 当前策略是否允许按 WRR 借用剩余带宽 |
| <code>qos_queue_pir_uncapped</code> | QoS 队列是否没有单独的 PIR 上限 |
| <code>qos_queue_defaults_gbps</code> | 各 group 中单条队列的默认 CIR |
| <code>total_requests</code> | 实际总请求数 |
| <code>completed_requests</code> | 完成请求数 |
| <code>makespan_ms</code> | 整场仿真的结束时间 |
| <code>events_processed</code> | 弹出的事件数，包括过期事件 |
| <code>stale_events</code> | 被版本号淘汰的过期事件数 |
| <code>event_counts</code> | 各事件类型计数 |
| <code>avg_ttft_ms</code> | 平均总 TTFT，包含排队 |
| <code>avg_processing_ttft_ms</code> | 平均处理 TTFT |
| <code>avg_queueing_delay_ms</code> | 平均排队时间 |
| <code>avg_npu_utilization</code> | 平均 NPU 计算利用率 |
| <code>throughput_requests_per_s</code> | 按 makespan 计算的吞吐量 |
| <code>batches_dispatched</code> | 总派发批次数 |
| <code>request_metrics</code> | 每条请求的详细统计 |
| <code>disk_stats</code> | 每块磁盘与每条队列的统计 |

每块磁盘的统计包括：

- 忙时间、空闲时间和利用率；
- 未使用带宽比例；
- 入队聚合流数量、对应的虚拟块数量、当前及峰值 outstanding 块数、带宽更新次数、令牌事件数、重分配事件数；
- 配置的 CIR 总和；
- 每条队列的 CIR、活动流上限、观测到的最大活动流数、最大分配带宽、服务量和激活次数。

NPU 利用率使用：

    npu.total_compute_ms / npu.compute_end_time

分母是从仿真时间 0 到该 NPU 最后完成时刻，因此包含请求排队、I/O 等待和没有任务的时间。

## 22. 使用示例

### 22.1 最基本调用

    import numpy as np

    from sim import load_bw_table_cache, simulate_continuous

    bw = load_bw_table_cache(num_npu=128)

    npus, summary = simulate_continuous(
        bw_table=bw,
        policy='queue_wrr',
        num_requests_per_npu=1,
        num_npu=128,
        num_disk=56,
        n_layers=16,
        rng=np.random.RandomState(42),
    )

    print(summary['avg_ttft_ms'])
    print(summary['avg_npu_utilization'])

### 22.2 分批派发

    from sim import IOSchedulingConfig

    io_sched = IOSchedulingConfig(
        io_dispatch_mode='batched',
        batch_size=8,
        batch_dispatch_interval_us=200.0,
        qos_queue_max_active_flows=4,
    )

    npus, summary = simulate_continuous(
        bw_table=bw,
        io_sched=io_sched,
    )

让 NPU 按每批实际数据量和请求需求带宽选择间隔：

    adaptive_io_sched = IOSchedulingConfig(
        io_dispatch_mode='batched',
        batch_size=8,
        batch_interval_mode='demand_aware',
        batch_dispatch_headroom=1.1,
        qos_queue_max_active_flows=4,
    )

### 22.3 多请求到达与两个实例

    npus, summary = simulate_continuous(
        bw_table=bw,
        num_npu=8,
        num_requests_per_npu=4,
        arrival_interval_ms=0.5,
        instance_config={
            0: [0, 1, 2, 3],
            1: [4, 5, 6, 7],
        },
        l1_strategy='least_loaded',
        l2_strategy='round_robin',
    )

<code>main.py</code> 已直接从 <code>sim.py</code> 导入实现。当前设置为 128 个 NPU、每个 NPU 一次推理（<code>N_REQ=1</code>）、16 层，使用 <code>SSU_LIST=[16, 28, 40, 56, 84, 112]</code>、<code>ls_ratio=0.5</code> 和 seed 42。普通运行包含 1 种 fair 模式和 5 种 queue_wrr 模式，共 6 × 1 × 6 = 36 次仿真。

当前实验的所有硬件队列都使用 active flow 上限 4，包括 fair 的唯一共享 path。queue_wrr 同时比较 all-at-once baseline、固定 200 µs 的 Batched(8)/TA(8)，以及 headroom=1.1 的自适应 Batched(8)/TA(8)。缓存会分别校验 fair、queue_wrr 和自适应间隔策略版本，避免复用旧批次语义的结果。

运行 24 队列三档配置：

    python main.py --qos-queues 24 --force

运行 256 队列、8 个等权 group 的配置：

    python main.py --qos-queues 256 --force

不加 <code>--force</code> 时会复用对应配置的完整缓存。两种队列数使用不同的缓存文件和图片后缀，不会彼此混用结果。

只测试 fair 时运行：

    python main.py --fair-only --force

它只保留 <code>baseline_fair</code>，按当前列表运行 6 次仿真。缓存按具体的 <code>(SSU 数量, ls_ratio, mode)</code> 组合检查；之后恢复普通运行时，会补跑缺少的 queue_wrr 组合，而不会误把仅含 fair 的缓存当成完整结果。

## 23. 自动测试覆盖了什么

运行：

    pytest -q

当前共 21 个参数化测试实例，覆盖：

1. 带宽表缓存缺失时能回退读取项目的 <code>data</code> 文件；
2. SSU 块数会排除 NPU 计算的 <code>nql</code> 个 token；
3. 不足 128 token 的最后一块使用真实大小，且所有块的数据量总和不变；
4. <code>nql</code> 不能超过请求的总 token 数；
5. 当 <code>nql</code> 等于总 token 数时，不会创建 SSU I/O 流；
6. 活跃 QoS 队列先得到 2.0、0.5、0.1 GB/s CIR，再按 4:2:1 借用剩余带宽；
7. PIR uncapped 允许单个活跃队列借满整块磁盘；
8. 256 队列会形成 8×32 的等权 group，使用 0.15625 GB/s CIR，并按 group 借用带宽；
9. 全部队列的 CIR 总和不能超过物理磁盘；
10. fair 在每个 SSU 上只有一条 path，最多 4 条流 active，其余按 FCFS 等待；
11. fair 聚合流按虚拟块数量加权，与未聚合流的竞争结果一致；
12. 可手算的 prefetch 结果为 2.5 ms；
13. 同一输入 sequential 结果为 3 ms；
14. NPU 忙时请求进入全局队列，释放后继续执行，并正确区分排队与处理 TTFT；
15. fair 与 queue_wrr 队列都只激活前 4 条流，并在完成后按 FCFS 补位；
16. batched 和 traffic-aware batched 都按 200 µs 间隔完成全部块；
17. 前一批传输超过 200 µs 时，后一批仍会发送并与它重叠；
18. 峰值 outstanding 块数和队列峰值 active flow 数被正确记录；
19. 只有 fair 结果的缓存只补跑缺少的 queue_wrr 组合；
20. 普通 batch 的自适应间隔等于“本批 GB / (required_bw × headroom)”；
21. traffic-aware batch 使用同一自适应公式，并在该时间点独立派发下一批。

仓库中的真实 <code>data</code> 表也已用于小规模烟雾测试，三种派发模式都能完成全部请求。

## 24. 性能优化与全量实测

当前实现的主要加速点是：

1. 同一 SSU、同一仿真时刻只做一次带宽重分配；
2. 等价块合并为带 <code>block_count</code> 的聚合流；
3. 活动流使用集合，并批量完成、批量移除；
4. traffic-aware 批次使用按 SSU 分组的双端队列和最小堆；
5. traffic-aware 压力使用增量维护的 outstanding 块计数，不逐次扫描 pending 队列；
6. 带宽重分配只遍历当前活跃的 QoS 队列，而不是每次扫描全部 24 或 256 条队列。

聚合是事件数量优化，不是近似带宽模型。小规模对照中，聚合前后的 makespan 都是 290.695687 ms；3 个虚拟块与 1 个虚拟块的公平分配也通过了单元测试。

在当前机器上，128 NPU、16 层、8 SSU、1 次推理/NPU、seed=42 的 fair 单点，优化前运行超过 176 秒仍未完成，优化后仿真程序的墙钟运行耗时约为 3.00 秒。这里的 3.00 秒是程序执行速度，不是 NPU 利用率；该点的 NPU 利用率是 30.029%。完整 fair 扫描共 21 个组合，墙钟运行耗时 92.36 秒，峰值常驻内存约 936 MiB。它们是 fair 全部 flow active 时的历史性能数据；当前 fair 已改为唯一 path 最多 active 4 条。时间也会随机器环境变化，这些数字主要用于展示优化量级。

下面 24.1 和 24.2 保留的是旧调度语义的历史对照数据：当时非 fair 队列深度为 1，而且下一批必须等待当前批全部完成。它们不能当作当前“active flow=4、独立定时发批”实现的结果；当前固定与自适应间隔对照实验使用带 <code>qd4_b200us</code> 的独立文件名。

### 24.1 历史：24 队列 PIR-uncapped 全量结果

完整的 24 队列实验共运行 84 次仿真，每个点取 3 个随机种子的平均值：

| SSU | fair | baseline_qwrr | batched8_qwrr | traffic_aware8_qwrr |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 29.3% | 29.3% | 32.0% | 33.2% |
| 16 | 44.3% | 45.0% | 47.5% | 49.0% |
| 28 | 57.1% | 59.1% | 60.4% | 62.7% |
| 40 | 65.9% | 68.2% | 68.4% | 71.6% |
| 56 | 74.4% | 76.6% | 75.9% | 80.5% |
| 84 | 85.5% | 85.9% | 84.5% | 90.1% |
| 112 | 91.1% | 91.8% | 89.6% | 94.5% |

旧的固定带宽实现中，8 SSU 的 <code>baseline_qwrr</code> 只有约 3.1% 利用率。改成 CIR 保底、空闲带宽可借用、PIR uncapped 后，同一点达到 29.3%，说明此前的低利用率主要来自未使用带宽被错误闲置。

当前机器完成这 84 次仿真的墙钟运行耗时约 3092.66 秒，峰值常驻内存约 2057 MiB。原始结果保存在 <code>results/io_scheduling_sweep_q24_pir_uncapped.pkl</code>，各模式曲线使用 <code>results/io_sched_16L_*_q24.png</code> 文件名。

### 24.2 历史：256 队列、8 个等权 group 全量结果

256 队列实验也完成了全部 84 次仿真：

| SSU | fair | baseline_qwrr | batched8_qwrr | traffic_aware8_qwrr |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 29.3% | 32.0% | 35.9% | 37.2% |
| 16 | 44.3% | 47.4% | 50.0% | 51.9% |
| 28 | 57.1% | 60.2% | 61.6% | 64.6% |
| 40 | 65.9% | 69.1% | 69.1% | 73.4% |
| 56 | 74.4% | 77.6% | 76.2% | 81.7% |
| 84 | 85.5% | 87.6% | 83.9% | 90.3% |
| 112 | 91.1% | 92.0% | 88.6% | 94.1% |

当前机器的墙钟运行耗时约 3094.37 秒，峰值常驻内存约 2097 MiB。原始结果保存在 <code>results/io_scheduling_sweep_q256_pir_uncapped.pkl</code>，各模式曲线使用 <code>results/io_sched_16L_*_q256.png</code> 文件名。

在低 SSU 数量、竞争较强的区域，256 队列配置比 24 队列配置更高。例如 8 SSU 时，baseline、batched、traffic-aware 分别提高约 2.7、3.9、4.0 个百分点。到 112 SSU 时，三者差值分别约为 +0.2、-1.0、-0.4 个百分点，说明 256 队列并不是所有模式和负载下都更好。

比较时还要注意：两个实验不仅队列数不同，分组、CIR 和借用权重也不同。因此这些结果表示“两套完整 QoS 配置”的差别，不能把全部差值只归因于队列数量。合并对比图保存在 <code>results/io_sched_16L_q24_vs_q256.png</code>。

### 24.3 当前：固定与 NPU 自适应发批的 q24 结果

当前缓存包含默认 <code>main.py</code> 的 36/36 个组合。配置为 128 NPU、16 层、每 NPU 一次推理、<code>ls_ratio=0.5</code>、seed 42；fair 的唯一共享 path 和每条 queue_wrr path 都最多 active 4 条。自适应模式使用 <code>interval = batch_gb / (required_bw × 1.1)</code>：

| SSU | fair | baseline_qwrr | Batch fixed | TA fixed | Batch adaptive | TA adaptive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 47.0% | 45.4% | 48.0% | 48.2% | 46.3% | 46.3% |
| 28 | 59.9% | 59.7% | 51.3% | 51.3% | 61.1% | 61.0% |
| 40 | 68.7% | 69.1% | 51.3% | 51.3% | 70.2% | 70.1% |
| 56 | 77.4% | 78.1% | 51.3% | 51.3% | 78.7% | 78.9% |
| 84 | 88.4% | 87.3% | 51.4% | 51.3% | 89.5% | 90.7% |
| 112 | 93.2% | 92.4% | 51.4% | 51.3% | 94.4% | 94.7% |

本次完整运行退出码为 0，墙钟耗时 1681.25 秒；写入完整缓存后再次运行只需 0.86 秒。缓存保存在 <code>results/io_scheduling_sweep_q24_qd4_b200us_pir_uncapped.pkl</code>，图片使用 <code>results/io_sched_16L_*_q24_qd4_b200us.png</code> 文件名。

固定间隔的两种分批模式几乎不随 SSU 数量提高，并不表示 batch 没有发生：测试已经确认后一批会在前一批未完成时进入同一队列，真实运行也处理了独立的 <code>BATCH_DISPATCH</code> 事件。主要原因是每个请求的一层只能每 200 µs 发 8 块；一层最后一批的最早发送时刻由块数和这个固定间隔决定，增加 SSU 不能缩短这段发送跨度。当该跨度大于实际传输瓶颈时，普通 batch 与 TA 都会被发送节奏主导，TA 的磁盘均衡收益也会被掩盖。

自适应模式消除了这个统一节奏瓶颈：从 SSU=28 开始，自适应 Batch 比固定 Batch 分别提高 9.8、18.9、27.3、38.1、43.1 个百分点，并保持接近或略高于 all-at-once baseline。SSU=16 时存储竞争最强，自适应 Batch 为 46.3%，低于固定 Batch 的 48.0%；更积极地提交并不保证在拥塞区也更好。TA adaptive 与普通 adaptive 大体接近，在 SSU=84 和 112 时分别达到 90.7% 和 94.7%，说明资源充足时按磁盘压力选块带来小幅额外收益。

## 25. 当前模型的边界和假设

理解结果时要注意：

1. QoS 队列的默认值是 CIR 保证，不是带宽硬上限；空闲带宽会按 WRR 权重借给活跃队列。
2. QoS 队列 PIR uncapped，但同一 SSU 上所有队列的带宽总和始终受 <code>disk_bw</code> 限制。
3. 24 队列配置的 QoS 层级由 <code>required_bw</code> 决定，不由 SS、SL、LS、LL 类别直接决定；256 队列配置不使用这三档阈值。
4. <code>NPU_BW_LIMIT</code> 当前用于理想时延参考，不会限制一个 NPU 跨多磁盘的实际聚合带宽。
5. 每块磁盘独立调度，没有模拟网络、PCIe、缓存命中、写 I/O 或磁盘内部寻址延迟。
6. fair 策略限制整块磁盘总带宽，但没有额外的 NPU 聚合带宽约束。
7. 块放置仍会创建大量描述字典。活动 I/O 已聚合为较少的流对象和事件，但 128 NPU、16 层的大实验仍可能描述上百万个虚拟块，内存占用不可忽略。
8. traffic-aware 批次只观察当前队列压力，是启发式算法。
9. 浮点完成时间使用很小的容差处理同一时刻事件。
10. <code>ttft_ideal_ms</code> 是参考输入，不直接决定实际事件完成时间。
11. 请求到达间隔目前是固定间隔；负载类型和随机放置由随机数生成器控制。
12. <code>placement_mode</code> 的未知值会回退到随机放置，实验代码最好显式使用受支持的名称。
13. 表中的 <code>per_layer_us</code> 是计算 nql 个 token 的逐层总时间，不会在仿真内再次按 nql 缩放。
14. 当前 24 或 256 条 QoS 队列都由多个 NPU/DPU 共享，并非每个 NPU/DPU 一条专属 path。
15. 256 队列的 8×32 分组和等权重来自用户给定配置；NPU、层到具体共享队列的均衡映射是本次实验假设，不是从硬件映射表读取的事实。
16. fair 只有一条共享 path，queue_wrr 有多条 QoS path；两者每条硬件队列都最多同时 active 4 条 flow，更多 flow 留在 FCFS pending 区。
17. 分批模式第 1 批立即发送，之后按固定间隔或 NPU 自适应间隔继续发送；两种方式都不等待上一批完成，因此允许 batch 重叠。
18. batch 重叠只增加 outstanding 队列压力；所有 flow 的 <code>demand_bw</code> 始终保持 data 表中的请求固定值，不会随批次等待而自动升档。
19. 自适应间隔策略使用本批数据量、请求 <code>required_bw</code> 和可调 headroom；它是对“NPU 能自主控制提交节奏”的当前模型，不代表已经复刻某款硬件的真实控制算法。最小/最大间隔只有获得硬件限制后才应设为对应数值。

## 26. 一句话总结

这套仿真器把一次请求拆成请求排队、NPU 分配、逐层 KV 块读取、QoS path 服务和逐层计算，再由事件堆按真实状态变化时刻推进；Batch(8) 每次发送整个 NPU 一层中的 8 个 128-token 块，既可固定 200 µs，也可由 NPU 按本批数据量和请求需求带宽选择下一批间隔，并允许批次重叠；所有硬件队列最多 active 4 条 flow；queue_wrr 先保证 CIR，再以 uncapped PIR 按权重借用剩余带宽，fair 则在每个 SSU 上只保留一条共享 path。
