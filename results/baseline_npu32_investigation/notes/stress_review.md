# Stress runner 独立只读审计

2026-09-07；审计者 `audit_lowutil`。未改动 `run_baseline_npu32_stress.py` 或 `sweep_baseline_npu32_stress.py`。

审阅版本：

```text
run_baseline_npu32_stress.py 94704edf908fbf90b9e39f5b26f94ad59c0c5dcfc4c1b981e1a2860921c2946d
sweep_baseline_npu32_stress.py 4a7b145c589d8588fd66df2404af35bcc497eb300fac02a6e69f733398dee56c
```

## 结论

没有发现会阻塞全新目录正式运行的仿真或统计错误。容量/Padding核算、同manifest跨策略配对、完整循环配额、角色与卡窗口统计均通过代码审阅及下列独立数值核对。

发现一个续跑复用风险，应在freeze前修复或通过全新输出目录、固定唯一variant配置规避：`sweep.execute` 见到 `runs/<input>/<variant>/*.json.gz` 已有结果时，只检查摘要invariants，没有核对当前job的strategy/config、frozen manifest fingerprint及源码哈希。相同variant更换参数后可以静默拿旧结果标记为existing。`compact` 随后还会使用新job的strategy/variant标签描述旧结果。此风险已立即通知root；本次不改源文件。

## 容量与归一化

`input_demand` 对每个原始NPU累计完整8层读量和理想计算时间，再求逐NPU逐SSU的GiB/s矩阵。单盘以40、单卡接收以50归一化，维度和单位正确。固定profile时结果等于ΣV/C；变profile时按时间加权，不能把单请求V/C算术平均当成相同指标。

`target_rho` 同时取最热SSU和最大NPU接收负载作为约束，只提高统一compute scale，不会为凑目标加速C。重建有限请求数量后再次精算实际manifest，最多8轮，未满足时报错。已数值检查目标0.5和0.95；前者得到最热SSU0.4999999995、最大接收0.4126240129，后者因原输入已低于目标保持scale=1，符合注释而非“总会精准达到rho”。

注意事项：

- `target_rho` 会改变C并重算请求数量，因此有/无归一化不是相同完整请求集；metadata已明确记录。不同策略共享同一冻结manifest则是严格配对。
- 每个原始NPU的有限平均需求是固定分配的必要资源检查；S1–S3重新选卡后，不能冒称此矩阵就是新分配的实际无stall需求。
- `local` 为 `group % num_ssu`。8个四卡组映射6/7盘时部分盘容纳两组，负载不均是实际构造结果；代码按真正最热盘归一化，不会误用8/SSU平均缩放。但报告要说明这不是均衡容量缩放。
- 程序允许rho最高2，因此过载组可合法构造；是否不足容量由`load_within_disk_and_link_capacity`判定。

## Exact 与 padded

Exact按真实剩余token计算尾块字节，padded仅将每个已有块补为176KiB，不改变块数量。每请求load里的`per_layer_kv_gb`和`required_bw_input_gbps`按实际placement字节重算，Padding总量为8层逐请求差额，方法正确。

独立构造4卡legacy、SSU1、horizon100ms、seed42：exact/padded均137请求，请求ID和C一一相同；exact rho=0.997153952610326，padded rho=0.9997591190541144；累计补齐0.060812950134277344GiB，与两个manifest逐字节差的8倍精确相等。Exact的`equal_176kib_blocks=False`，padded为True。原始格点画像无需Padding时全为完整块，native/shared可以共用。

旧1K short tail的补齐比例相对其本层较大，因此即使全体总字节差很小，也必须做exact native↔padded native桥接；不能用整体补齐很少就宣称受害者没有变化。`aligned`改变NQL，是另一组画像，不是对旧exact单纯换格式。

## 同输入与策略对照

冻结manifest保存placement和load，读回重建后核对完整input fingerprint。`run_case`执行前后也核对输入fingerprint不变。Coflow原runner另外核对执行placement未变，即允许选卡但保持数据落盘。Submit seed来自同一metadata。

Native baseline/Once使用`pressure_ttl_ms=0`；shared baseline/Once/New Once和S1–S3使用真实5ms collector。Native deadline族保留其独占Path/CIR控制；S1–S3还可按pipeline重分配NPU，fixed消融才隔离调度。同输入对比合法，但应在图表中写清各策略信息与动作范围不同。

所有请求在t=0到达，输入是有限饱和backlog。S1–S3在到达时选卡可看到当时已经到达的工作；这不是生产外部218GiB/s输入，更不能把所有t=0算作稳定入口速率。Arrival SLO会包含特意制造的巨大admission backlog，代码保留它是正确的，但应与admission SLO分开解释。

## 完整配额与同步/顺序对照

`synchronized`和`shuffled`都按完整循环数量终止，每个NPU每个role次数相等，整个输入总C和总字节严格相同。独立4卡raw构造(`32:512,32:512,192:1024,192:1024`，horizon100ms)两种order均32请求；每卡4种role各2次，理想C均1209.693258284553ms。

需要保留三个解释限制：

1. Fixed按每卡固定role的C达到同一计算horizon，因不同C请求次数不同；shuffled按所有角色等次数循环。Fixed→shuffled改变每卡画像、全体role配额和总平均需求，不能称只打乱同一批请求顺序。Synchronized↔shuffled才是完整配额匹配的顺序控制。
2. Synchronized意味着各卡拿到相同profile序列，不保证I/O反馈后release始终同相；shuffled每卡只抽一个deck并循环复用，不是每周期重新随机，也不是独立逐请求随机到达。
3. Hash placement依赖request_id；换order把不同大小profile映射到不同ID，会轻微改变逐盘字节。独立2盘小测，相同配额synchronized盘率16.15732750628621/15.455800233410912，shuffled变16.074076643207658/15.539051096489468GiB/s。Local和stripe同配额对照逐盘率完全相同。若希望最严格隔离顺序影响，优先local/stripe；hash组须承认落盘差异。

## 窗口、角色和stall核对

固定窗口通过每个完整microbatch接纳→完成区间求active交集、逐层compute求交集；逐卡idle=duration-active、stall=active-compute。角色由稳定request_id查回原profile role，故NPU重分配不会把角色归错。当前batch=1，角色归属无歧义。

Warm层只取L1–L7且release位于窗口内；latency为预取逻辑开始到HBM-ready，budget取前层真实计算时长，slack=budget-latency。Coflow延迟实际发命令时，逻辑start仍保留激活时刻，因此该指标包括客户端延迟，没有把调度等待隐藏掉。L0未进入warm统计，但其暴露等待仍进入active-compute，这是有意不同的样本定义。

读取5个已保存pilot完整summary（aligned的baseline/native baseline/native Once/S3，以及legacy native baseline），独立重算：

- role compute/active/stall总和与逐卡窗口值完全相等，差为0。
- 所有暖层`io_start == 前层compute_start`成立；`max(0,-slack) == io_barrier_wait`最大数值误差小于5e-14ms。
- Aligned四策略的输入fingerprint全部为`87c0bcd8b4979ff90502e2c5393af48e2d20da90a5e6929c67400e849cea3b09`。
- 全部已查summary invariants通过。

`all_npus_active_whole_window`会如实标识pipeline重分卡后部分卡提前排空；runner没有强制拒绝这些结果，因此正式报告必须保留idle标记，而不能全部称为饱和I/O stall样本。Warm层release窗口样本与完成/接纳窗口样本不是同一总体；报告若使用其百分位应写清采样边界。完整请求SLO函数检查所有输入ID都已完成，不存在窗口末尾删掉未完成请求的问题。

## Sweep 运行审阅

每个job用独立子进程，避免adapter monkey-patch跨策略污染；进程有超时，状态每45秒写入，完成结果即时收集。每个仿真自带核心hash检查，sweep再检查所有启动时根目录Python文件未变化，适合freeze后运行。

除前述旧结果复用问题，还需确保plan中`(input,variant)`唯一，避免同目录并发写入不同配置并导致结果歧义。新输出目录、不可修改的plan和唯一variant足以规避现有风险。Source hash不包含目录之外新增代码的全面快照保证，但本任务原核心文件均已保存hash和初始tar；这不阻碍当前实验复现。
