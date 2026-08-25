# 静态 CIR QoS 与 Baseline Bypass 仿真

本项目现在只保留两种策略：

- `baseline_bypass`：绕过 QoS Path，每个 NPU 内部保持 FCFS，NPU 之间逐 I/O RR。
- `qos_static_cir`：SSD 初始化时只写入一次 CIR、PIR、Path WRR 和 Group WRR；运行期间不再修改这些硬件配置。默认每 8 个 I/O 读取一次压力，但每个 I/O 独立选择 QoS Path。

旧的动态 K/q、layer-global refill、动态硬件配置、token bucket 和 Path 租约已经移除。固定的 8-I/O 客户端提交节奏仍保留，用于模拟同一时刻按 NPU 随机轮转提交；它不是自适应 batch 策略。硬件侧的 8 个 Group、256 个 Path、静态 CIR/PIR，以及组内 WRR + 组间 WRR + 最终 RR 仲裁仍然保留。

## 统一的单后端 I/O 模型

两种策略共享完全相同的物理约束：一块 SSD 在任意时刻最多只有一个不可抢占的 active I/O，完成后才重新仲裁；同一个 NPU 从所有 SSD 返回数据的**有效带宽合计始终不超过 50 GB/s**。SSD 侧 raw 速率最高为 40 GB/s；若一个 NPU 同时得到两路 40 GB/s，数据面按比例变成 25+25 GB/s，释放的 credit 会立即供该 NPU 的其他 active I/O 使用。

```text
baseline:  各 NPU FCFS 队列 --逐 I/O RR------------------> 唯一后端槽位
QoS:       256 个 Path FCFS 队列 --CIR/WRR/RR 选一 Path--> 唯一后端槽位
```

QoS 中的 CIR/PIR/WRR 不再表示多个 active I/O 同时分割连续带宽；它们决定各 Path 在长期内获得离散调度机会的比例。

## QoS I/O 是怎样提交的

```text
一层 I/O 按目标 SSU 拆成独立提交状态
        ↓
全局按 ready_time 调度；同一时间随机排列 NPU，每轮每个 NPU 最多提交一个 batch
        ↓
每个提交 batch 最多 8 个 I/O；下一轮重新随机排列
        ↓
按 pressure_read_interval 切压力窗口（默认 8），窗口开始时读取 256-Path 压力
        ↓
客户端取得当前请求类别的静态 Path 池
        ↓
客户端估算每个候选 Path 所在 Group 的竞争和长期等效服务率
        ↓
窗口内按 path_binding_batch_size 切绑定批次（默认 1，即逐 I/O）
        ↓
每个绑定批次选择“加入后，预计最早清空积压”的 Path，并更新本地影子压力
        ↓
SSD 按指定 Path 入队，由 CIR + 两级 WRR + RR 选出一个 Path 队头
```

必须分清两个概念：

- `pressure_read_interval` 控制遥测读取频率；`0` 表示当前 `(request, layer, SSU)` 的全部 I/O 只读取一次，正整数 `N` 表示每 `N` 个 I/O 读取一次。
- `path_binding_batch_size` 控制选路粒度；`1` 表示每个 I/O 独立选路，`N` 表示连续最多 `N` 个 I/O 强制共用一个 Path。绑定批次在每个压力窗口内重新开始。
- `client_submit_batch_size` 控制一次客户端事件最多提交多少个 I/O，默认 8；Baseline 与 QoS 使用同一个值。
- `client_submit_interval_us` 控制同一 NPU 两次 batch 的最小发送间隔，默认 0；即使为 0，不同 NPU 仍按随机轮次交错提交。
- `submit_order_seed` 只控制同一 `ready_time` 的 NPU 随机排列，使用独立随机源，不改变请求抽样或 block 放置。
- 客户端可以选择 `io_count > 0` 的 Path。没有“只有空 Path 才能使用”的限制，也没有 Path 所有权。
- 仿真中的每个 Path 使用先入先出队列；只有当该 Path 被整盘仲裁选中时，队头 I/O 才会成为 active。全盘 256 个 Path 合计最多一个 active I/O。这是仿真模型，不是对真实 SSD 队列深度的硬件结论。

## 静态 Path 布局

每块 SSD 有 8 个 Group，每组 32 个 Path。组内四类 Path 的固定布局为：

| 类别 | 每组 Path 数 | 全盘 Path 数 | 全盘静态 CIR 预算 |
|---|---:|---:|---:|
| SS | 12 | 96 | 20 GB/s |
| SL | 4 | 32 | 4 GB/s |
| LS | 12 | 96 | 12 GB/s |
| LL | 4 | 32 | 4 GB/s |

客户端仍然只在本类别的 Path 池中选路，不跨类别。`client_category_paths()` 返回的池按 Group 交错排列；例如 LL 池开头为：

```text
28, 60, 92, 124, 156, 188, 220, 252, 29, 61, ...
```

因此所有候选完全相同时，连续几个独立选路的 I/O 会优先分散到不同 Group，而不是先堆到 Group 0。只有同一个 Path 绑定批次内部才会强制使用一个 Path。

## 可直接移植的客户端选路

真正的选路集中在 `sim.py` 的 `client_select_qos_paths()`。它是一个只有 4 个参数的纯函数：不读取仿真全局状态，不提交 I/O，不修改压力报告，也不修改 CIR/PIR/WRR。相同输入一定得到相同输出。最小移植单元是这个函数、`ClientRoutingConfig` 和 `StaticQoSConfig`；其他项目也可以用字段相同的只读结构替换两个数据类。

### 四个输入

```python
path_ids = client_select_qos_paths(
    block_sizes_gb=block_sizes_gb,
    path_io_counts=pressure_report,
    allowed_path_ids=allowed_paths,
    routing_config=routing_config,
)
```

1. `block_sizes_gb`

   发往同一块 SSD 的数据块大小列表，单位为 GB。返回结果的第 `i` 项与这里的第 `i` 个数据块对应。

2. `path_io_counts`

   SSD 最新汇报的完整 256 项压力快照。数组下标是 Path ID，值是该 Path 的 `active + pending` I/O 数。必须传完整报告，因为候选 Path 还会受到同 Group 中其他类别 Path 的影响。

3. `allowed_path_ids`

   当前请求类别允许使用的 Path ID。函数绝不会返回这个集合以外的 Path。

4. `routing_config`

   一个只读 `ClientRoutingConfig`，包含：

   - `qos_config`：客户端初始化时缓存的静态 CIR/PIR、Path 权重、Group 权重和布局镜像；
   - `disk_bw`：目标 SSD 的物理带宽，单位为 GB/s；
   - `start_offset`：所有分数完全相同时的轮转起点，它是候选数组下标，不是 Path ID；
   - `path_binding_batch_size`：连续多少个 I/O 强制共用同一个 Path，默认值为 1，即逐 I/O 选路。这个纯函数只消费一份已有压力快照，不负责决定快照读取频率。

纯函数会检查 256-count 长度、非负整数、合法 Path 池和 block 大小；静态配置的数据结构完整性在 `StaticQoSConfig` 初始化时检查。

### 输出

输出是一个 Path ID 列表，长度与 `block_sizes_gb` 相同。调用方把第 `i` 个 Path ID 写入第 `i` 个 I/O 的 SQE DW2，然后向已经选定的 SSD 提交它。

### 最小示例

```python
from experiment import qos_config
from sim import (
    ClientRoutingConfig,
    client_category_paths,
    client_select_qos_paths,
)

# 这些静态信息在客户端初始化时准备一次即可。
static_config = qos_config()
allowed_paths = client_category_paths("LL", static_config)

# 这 8 个数据块已经由上游放置模块确定要发送到同一块 SSD。
block_sizes_gb = [0.001] * 8

# 读取这块 SSD 最新汇报的全部 256 个 Path 压力。
pressure_report = [0] * 256

# 这里只构造客户端的只读估算参数，不会写 SSD 寄存器。
routing_config = ClientRoutingConfig(
    qos_config=static_config,
    disk_bw=40.0,
    start_offset=0,
    path_binding_batch_size=1,
)

# 四参数纯函数只计算 Path ID，不执行 I/O。
path_ids = client_select_qos_paths(
    block_sizes_gb=block_sizes_gb,
    path_io_counts=pressure_report,
    allowed_path_ids=allowed_paths,
    routing_config=routing_config,
)

# 只读取了一次压力，但函数会更新本地影子压力并为 8 个 I/O 分别选路。
assert len(path_ids) == 8
assert len(set(path_ids)) == 8

for block_size_gb, path_id in zip(block_sizes_gb, path_ids):
    # 真实客户端在这里把 path_id 写入 SQE DW2，再提交这个数据块 I/O。
    print(block_size_gb, path_id)
```

## Group 感知 SED 算法

SED 在这里表示“最短预计完成时间”。函数为一个 I/O 批次检查候选 Path 时，执行以下计算：

1. 用完整压力报告找出所有活跃 Path 和活跃 Group。
2. 按硬件规则估算候选入队后的长期等效 Path 服务率：
   - 活跃 Path 先获得 `min(CIR, PIR)`；空 Path 未使用的 CIR 回收到整盘公共池；
   - 剩余物理带宽按活跃 Group 的 WRR 权重分配；
   - Group 得到的剩余带宽再按组内活跃 Path 的 WRR 权重分配。
3. 用下面的主分数比较候选：

   ```text
   预计清空时间 =（候选 Path 的预计积压数据量 + 当前批次总字节数）
                  / 候选 Path 的长期等效服务率
   ```

4. 分数相同时，依次选择：预计 I/O 数更少、Group 总 I/O 数更少、轮转顺序更靠前的 Path。
5. 选中后按照批次内的真实 I/O 数和总字节数增加 Path 与 Group 的本地影子压力；同一批的所有 I/O 返回同一个 Path ID。
6. 如果一次纯函数调用包含多个批次，函数先规划总字节数较大的批次；最终仍按输入顺序返回 Path ID。

这比旧的“只选择最小 `io_count`”多看了两类信息：

- Group 压力：两个候选 Path 的自身 `io_count` 相同，繁忙 Group 中的 Path 通常仍会获得更少的可借用带宽。
- I/O 大小：一个大数据块和一个小数据块不再被当成完全相同的工作量。

硬件目前只汇报 I/O 数，没有汇报每个旧 I/O 的剩余字节。因此函数使用“本批数据块的中位大小 × 汇报 I/O 数”估算旧积压。这意味着它是完成时间估计，不是对未来完成时刻的精确预言。

当前实验配置中所有 PIR 都是无限值，Group/Path 权重都为 1。选择器使用与 SSD 离散调度器相同的长期服务权公式，用于估算队列清空时间；它不表示多个 I/O 会同时获得这些带宽。如果未来配置有限 PIR，还需要补充其他 Path 触顶后的二次剩余服务权再分配。

## 硬件侧保持不变

`StaticQoSConfig` 是 frozen dataclass，只包含：

- `path_cirs`
- `path_pirs`
- `path_weights`
- `group_weights`
- `category_paths_per_group`

每块 SSD 创建 `DiskIOScheduler` 时，用这些值建立 256 个 `PathQueue`。运行期间变化的是队列压力、虚拟服务标签和当前唯一的后端 I/O，不是 CIR/PIR/WRR 寄存器。

`DiskIOScheduler.report_path_io_counts()` 是 QoS 硬件到客户端的唯一动态接口。`pressure_read_interval=N` 表示同一 `(request, layer, SSU)` 内每 N 个 I/O 读取一次报告，尾窗口不足 N 个也读取一次；`pressure_read_interval=0` 表示首次提交时读取一次，并为这组全部 I/O 提前规划 Path。窗口内每选完一个绑定批次，客户端都会更新本地影子压力。后续实际提交按 8-I/O batch 与其他 NPU 交错：`0` 不会看到其他 NPU 后来的占用，正整数 `N` 到达下一压力窗口时会重新读取。`path_binding_batch_size` 只控制连续多少个 I/O 共用 Path，不会触发额外硬件读取。当前仿真假设报告零延迟、无丢失；真实系统若周期上报，还需要另外建模按时间更新的陈旧快照。

## 运行 16 层实验

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiment.py \
  --layers 16 \
  --ssu-list 40,56 \
  --workers 2 \
  --rerun \
  --output-dir results/routing_comparison_capped_v3
```

该入口固定运行严格配对的四组：

- `capped baseline`：QoS bypass；
- `0/1`：每个 `(request, layer, SSU)` 只读一次压力，每个 I/O 独立选 Path；
- `8/1`：每 8 个 I/O 读一次压力，每个 I/O 独立选 Path；
- `8/8`：每 8 个 I/O 读一次压力，并让这 8 个 I/O 共用 Path。

实验固定使用 128 个 NPU、`LS_RATIO=0.5`、seed 42。schema v3 把
`npu_aggregate_cap_gbps=50` 和代码/data 指纹写入 cache spec，不会复用旧的
uncapped 结果。历史 `results/routing_comparison_l16` 的 79%/85% 等数字没有
NPU 50 GB/s 聚合上限，只能作为历史参考，不能与当前结果混用。

当前 16 层实测中，40/56 SSU 的 capped baseline 为 61.76%/64.65%；最佳 QoS
变体 8/8 仅为 46.08%/39.60%，0/1 和 8/1 更低，并且全部违反类别 p95 与
fairness 门槛。完整指标与原因分析见
[capped 16 层结果](results/routing_comparison_capped_v3/RESULTS_CN.md)。

## 测试

```bash
MPLCONFIGDIR=/tmp/qos-demo-mpl python -m unittest discover -s tests -v
```

35 项测试覆盖静态 CIR 布局、配置不可变、严格类别池、非空 Path 可选、压力读取与 Path 绑定解耦、ready-time 随机轮转、256 项 active+pending ABI、Path FCFS、全盘最多一个 active I/O、Baseline RR、QoS 的 CIR/两级 WRR/RR、客户端快速选路与参考算法等价、heap 仲裁与全扫描逐步等价、block/placement/bytes 守恒，以及两种策略共同遵守 NPU 50 GB/s 上限。

## 文件

- `sim.py`：离散事件引擎、静态 QoS 硬件模型，以及可直接移植的四参数客户端选路函数。
- `experiment.py`：capped Baseline 与 0/1、8/1、8/8 的唯一实验入口和绘图。
- `tests/test_capped_simulation.py`：NPU cap、优化等价性和守恒集成测试。
- `tests/test_experiment.py`：核心行为回归测试。
- `data`：请求带宽与时延输入表。
