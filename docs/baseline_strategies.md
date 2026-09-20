# ASU 与 OD 基线

两种基线只改变SSU上的Path使用方式和静态QoS配置，不改变输入请求、每卡请求顺序、执行卡绑定或数据块的落盘位置。

| 策略 | 每盘可路由的Path | NPU如何使用Path | 带宽配置 |
|---|---|---|---|
| `asu_baseline` | 1条，Path0 | 所有NPU共用Path0，保持原FIFO行为 | 保留原静态QoS表；唯一有请求的Path可使用整盘服务机会 |
| `od_baseline` | 与NPU数相同 | 每张NPU独占一条Path，同一NPU跨请求、跨层保持绑定 | 每条独占Path的CIR为`盘带宽/NPU数`，允许借用空闲带宽 |

`baseline`是`asu_baseline`的兼容别名，旧实验脚本仍可调用。历史结果中的`baseline`名称和数值保留原样。两个共享Path实验入口的默认策略已改为`asu_baseline`；通用压力脚本还保留旧的`baseline_native`原生回放入口，它不是OD。

## OD的独占Path映射

当前硬件模型有8个Group，每组32条Path，共256条硬件寄存器。OD将独占Path均匀分布到这些Group：

```text
path_id = (npu_id % 8) * 32 + npu_id // 8
```

| NPU序号 | 每张SSU上的Path序号 |
|---|---:|
| 0 | 0 |
| 1 | 32 |
| 7 | 224 |
| 8 | 1 |
| 31 | 227 |

32张NPU时，每个Group使用4条Path，每盘共使用32条；其余224条Path的CIR和剩余带宽权重均为0，且不会被选路。底层仍有256条硬件寄存器，因此“启用32条”指32条可路由的独占Path，不是把硬件数组缩成32项。实现支持1至256张NPU。

绑定以实际执行NPU为准，不取请求中可能保留的原始来源卡标记。下一请求的首层预取也使用该执行NPU的独占Path。Path内仍是FIFO，盘每次只执行一个不可抢占的I/O命令，没有请求重排、运行时CIR更新或动态迁移。

## “带宽均分”的准确含义

32张NPU、每盘40 GiB/s时：

```text
每条独占Path的CIR = 40 / 32 = 1.25 GiB/s
3张SSU合计的每卡保证份额 = 3 * 1.25 = 3.75 GiB/s
```

这是**保证份额CIR均分**。PIR未设硬上限：空闲卡没有用完的服务机会可以被有请求的卡借用。全部32条Path持续积压时，它们的长期服务份额相同；只有一条Path有请求时，它可以使用整盘40 GiB/s。由于盘按完整I/O命令串行服务，“份额”不是每个瞬间同时传输32路数据。

借用沿用原硬件的两级WRR：先按权重分给有请求的Group，再在组内分给有请求的Path。不同Group内活跃Path数量不同时，各活跃NPU获得的**总带宽不一定相等**。因此不能把OD描述成“任意时刻都给每卡同样的实际带宽”，也不能把1.25 GiB/s写成每卡每盘无法超过的限速。

OD不读取请求deadline来分配CIR，也不按每层需求`B=V/C`分配份额。独占Path消除了不同NPU共用一条FIFO队列的方式，但各卡仍竞争同一物理盘；读取必须等所需各盘完成，计算才能开始。均分CIR不保证每种请求的利用率或SLO都改善。

## 通用入口

从项目根目录执行。下面两条命令使用32张NPU、6张SSU的小输入，目的是检查入口；`--small`不是正式warm窗口实验。输出位置另行指定，避免写入历史结果目录。

```bash
python -m inputs.runners.run_shared_path_experiments \
  --num-npu 32 --num-ssu 6 --seed 20260906 \
  --strategy asu_baseline --small \
  --output /tmp/qos_asu_shared_smoke

python -m inputs.runners.run_coflow_experiments \
  --num-npu 32 --num-ssu 6 --seed 20260906 \
  --strategy od_baseline --assignment fixed --small \
  --output /tmp/qos_od_coflow_smoke
```

两个入口均接受`asu_baseline`和`od_baseline`，省略`--strategy`时默认ASU。它们各自的命令行输入生成器目前只接受5、6、7张SSU；**3张SSU的多样画像实验使用下面的专用入口**。文件已存在时运行器会拒绝覆盖。修改策略后应使用相同输入参数进行配对比较。

## 32 NPU、3 SSU的多样画像实验

新实验目录为[od_baseline_diverse_ssu3_20260918](../results/od_baseline_diverse_ssu3_20260918/README.md)。输入继承此前多样data实验的画像、配比与Random顺序，落盘统一为Ring hash；每个场景和种子共用冻结manifest，再分别运行各策略。旧条带实验的统计不能直接充当新Ring hash输入的ASU对照。

运行单个完整case：

```bash
python results/od_baseline_diverse_ssu3_20260918/run_trial.py \
  --manifest results/od_baseline_diverse_ssu3_20260918/inputs/full_seed7_ring_hash.json.gz \
  --policy od_baseline \
  --label manual_full_od_seed7
```

`--label`决定该实验目录下的`runs/<label>`，必须是尚不存在的新目录名。将`--policy`改为`asu_baseline`、`once`、`static`、`mild`或`aggressive`可运行对应策略；每次使用独立label。`semi_seed7_ring_hash.json.gz`对应间歇过载输入，`full_seed7_ring_hash.json.gz`对应持续过载输入；另外保留seed 19、43。

默认会完成全部有限输入。`--smoke`只取每卡首个请求，不能作为正式统计结果；`--pilot`在warm接纳集合及所需服务完成后停止，也不能当成完整仿真结果。正式实验不要使用这两个选项。

主窗口仍为`[2,4)`秒：NPU利用率按窗口内实际计算卡时间统计；SLO按该窗口内接纳的请求统计，并跟踪至各自完成，阈值为其纯计算时间的1.5倍。这里的完成耗时不含接纳前排队，不是真实端到端首token时间。不同策略的warm接纳集合可能不同，报告同时保留完整输入及其他窗口的检查。

## 配置与路由审计

核心结果提供`canonical_strategy`、`baseline_configuration`、`static_path_cirs_gib_s`和源码SHA。`adapter_statistics.routed_blocks_by_ssu_npu_path`统计ASU/OD逐盘、逐NPU、逐Path的规划块数。专用实验运行器还从实际完成I/O检查独占归属，并核验实际安装的QoS表，避免仅凭策略名推断配置。

相关实现：[静态基线配置](../simulator/policies/baselines.py)、[执行适配器](../simulator/adapters/shared_path.py)、[基线测试](../tests/test_od_baseline.py)。
