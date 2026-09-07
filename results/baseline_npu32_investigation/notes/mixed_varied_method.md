# 每卡多画像混合输入补充实验

这组实验回应“每张卡固定重复一种画像不实际”的修正：每张 NPU 都包含全部 4 或 5 种画像，其中 3 种属于短请求。每卡配额完全相同，先构造足够长的完整请求人口，再用 `random.Random(seed+npu_id*100003).shuffle` 对每卡整条列表各洗牌一次。不会循环重放一个已排好序的小 deck。每个配额运行 seed 7 和 123，全部在观察结果前固定；各输入的 Baseline、Once、New-Once 在同一台主机上运行。

这里的 `role=short` 按相对于 192K 长画像的计算时间定义，并不保证都是模拟器的 `category=SS`。`sim.classify_request` 的两个维度是序列长度是否 ≤80K、NQL 是否 ≥512，因此 SS 不直接等价于计算时间短。例如 `(32,512)`、`(64,512)` 属于 SL；aligned 的长画像 `(192,256)` 属于 LS，raw 的两长画像属于 LL。画像角色与调度器路径分类必须分别报告。为复核“短流”严格指 SS 的解释，另建 [mixed_strict_ss](../mixed_strict_ss/plan.json)：前三画像 `(32,128),(48,256),(64,128)` 全部属于 SS，后两 `(192,1024),(192,2048)` 属于 LL。该额外 2 seed × 3 策略 = 6 次模拟单列，不混入本文 36 次主计划；实际分类证据见 [semantic_audit.json](../mixed_strict_ss/semantic_audit.json)。

构造器为 [run_mixed_sustained_probe.py](../run_mixed_sustained_probe.py)，冻结 SHA256 为 `979028c9117774c3cd5d9a996b972da070002e24b7a7e77e94c9ed8ab717f00b`。正式人口与指纹在 [plan.json](../mixed_varied/plan.json)，配额定义在 [selected_spec.json](../mixed_varied/selected_spec.json)，逐请求输入在 `mixed_varied/inputs/*.json.gz`。所有模拟均从 manifest 加载，远端不重新抽样。独立审计脚本和结果位于 [aggregate_mixed_profiles.py](aggregate_mixed_profiles.py) 与 `mixed_profile_audit/`。

| 家族 | 每卡配额单元 | 配额单元数 | 短请求理想计算时间占比 | 每卡理想计算 ms | 最高 SSU 负载 | 最高 NPU 接收负载 |
|---|---|---:|---:|---:|---:|---:|
| aligned_short072 | 94:94:94:9 | 2 | 71.8164% | 4449.4168 | 0.935417 | 0.187083 |
| aligned_short080 | 65:65:65:4 | 3 | 79.8573% | 4150.3934 | 0.709111 | 0.141822 |
| raw_equal_rho090 | 17:17:17:1:3 | 2 | 32.9176% | 5683.0271 | 0.899037 | 0.179807 |
| raw_equal_rho097 | 21:21:21:1:3 | 2 | 37.7399% | 6123.1959 | 0.967915 | 0.193583 |
| raw_skew_rho096 | 13:13:104:3:3 | 1 | 58.1766% | 5862.1942 | 0.961327 | 0.192265 |
| raw_size_varied_rho097 | 14:14:14:1:3 | 2 | 37.5494% | 6104.5197 | 0.973032 | 0.194606 |

前两组的画像顺序为 `(1,128),(1,256),(1,384),(192,256)`，1K 画像沿用明确记录的外推构造。中间三组直接取数据表 `(32,128),(32,256),(32,512),(192,1024),(192,2048)`。最后一组直接取 `(32,128),(48,256),(64,512),(192,1024),(192,2048)`，短请求同时改变计算时间、读取体积和序列长度，避免只有 NQL 变化而体积近似不变。

全部使用 32 NPU、8 SSU；每层 176 KiB 整块，保持画像计算时间与读取体积不变，不加 padding。块条带落盘规则为 `ssu=(block_index+original_npu_id//4)%8`。所有请求在 t=0 到达，Baseline/Once/New-Once 保持原始 NPU 绑定，收集周期 5 ms。每卡理想计算工作量至少 `4000 ms + 2×最大完整请求计算时间`，主要观察窗口预先固定为 [1000,2000] 与 [2000,3000] ms；结果仍需单独核验这些窗口是否全部 NPU 持续有工作。

表内负载是实际冻结人口的计算时间加权全量字节需求，相对于每盘 40 GiB/s、每卡接收 50 GiB/s 的比值。这是长期容量的必要条件，不能证明每一个短期读取期限都可满足。配额不同会改变人口，不能把两行视为仅改变一个时间顺序的反事实。它们也不是先前约 52% 利用率固定角色输入的同一人口。

计划共有 12 个输入 × 3 策略 = 36 次模拟。`local/` 与 `remote/` 各 15 次、4 workers；新增体积变化组在 `extra_local/` 与 `extra_remote/` 各 3 次、3 workers。每个 job 的 `command.json` 保存命令、输入指纹、环境、进程号、开始结束 UTC、返回码及墙钟耗时。2026-09-06 17:01 UTC 左右开始主批。远端主 SSH 工具会话曾返回 255，但核实原 orchestrator 与 4 个子进程仍存活，未重启；基础设施事件记录在 [transport_incidents.json](../mixed_varied/transport_incidents.json)，不计为策略失败。

主计划现已完成唯一 36/36 次模拟，所有命令返回码为 0，全部输入指纹、实际 NPU 绑定、源码哈希、模拟器 invariants 与双观察窗 all-active 核验通过。[最终采集核验](../mixed_varied/collection_status.json)。远端原始 2/15 状态保留在 `remote/run_status_transport_original.json`，当前 `remote/run_status.json` 由已完成结果和 command 记录重建为 15/15。

必须同时报告总体利用率与同一完整画像 cohort 的短请求 stall/SLO。总体高利用率可能由长计算时间占比支撑；整体下降也不自动意味着全部短请求类型都同样受损。全部策略完成前，不把尚未完成的结果推断为策略改进或退化。输入频率、t=0 积压和独立洗牌仍是受控合成条件，不能称为生产流量实测。

另一个可以直接算出的边界：若最新 raw 组只保留 `(32,128),(48,256),(64,512)` 三类短请求、每卡等请求数，删除两类长请求，池化需求为每卡 18.7807015 GiB/s，32 卡/8 盘的磁盘负载为 1.87807015。仅由完整人口总字节数即可推出 full-run `total_compute/(32*makespan) ≤ 53.2461%`。这不是任意中间窗口的利用率上界，也不是原混合输入的模拟结果；它说明把输入变为“全是这些短请求”会同时造成物理带宽过载，任何调度器都不能消除总量不足。推导和原始 C/V 在 [raw_varied_allshort_capacity.json](raw_varied_allshort_capacity.json)。
