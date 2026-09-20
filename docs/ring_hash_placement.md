# 当前 placement：仅使用 Ring hash

2026-09-17 起，新输入的生成代码只保留 `sim.block_ring_hash_disk_id`。已删除 `local`、按组偏移 `stripe`、`hotspot`，并将 A/B、多样 data、near-capacity 及其构造画像生成器中的按 NPU 取模条带改为 Ring hash。

```text
SSU = block_ring_hash_disk_id(稳定请求身份, block_index, SSU数量)
```

实现位于 [`simulator/core/sim.py`](../simulator/core/sim.py)：使用 SHA-256 一致性哈希环，每盘256个虚拟节点。同一数据块的哈希键不含层号，因此各层复用其落盘位置。Ring hash 不保证每个请求在各盘上的块数严格相等。

## 请求重排不改变物理位置

普通生成器的稳定身份是 `request_id`。A/B、diverse 和 near-capacity 生成器用队列位置作为执行时的 `request_id`，另以 `original_request_id` 保存重排前身份，因此这些生成器哈希 `original_request_id`。元数据 `placement_identity_key` 明确记录使用哪个字段。

在同一批请求内，只改变 Random/Ordered 或 shuffle seed，不会把已有请求的数据重新分配到另一盘。改变请求人口、画像、盘数属于另一个输入配置，不能据此要求所有位置或带宽统计保持不变。

每个请求分别生成 hash placement，不能再按“同一NPU、同一画像”共用一份条带模板。每盘平均需求按实际落盘量计算：

```text
单卡对某盘的理想平均需求 = 该卡所有请求每层读取该盘的总量 / 所有请求每层计算时间之和
某盘整机平均需求 = 各卡对该盘理想平均需求之和
某盘静态需求上界 = 各卡该盘最大单请求(V/C)之和
```

这里是完整人口的理想需求及静态边界，仍然不等于仿真某一时刻的请求需求或实际供给。

## 运行入口

通用压力脚本默认 Ring hash，兼容的显式选项只有 `--layout hash`；其他值会报错，`--hotspot-fraction` 已删除。例如只生成输入：

```bash
python -m inputs.runners.run_baseline_npu32_stress \
  --family raw --raw-keys 128:256,32:4096 --num-npu 32 --num-ssu 3 \
  --layout hash --blocks exact --describe-only \
  --manifest-out /tmp/qos_ring_hash_input.json.gz
```

各研究目录生成器的新默认标签带 `ring_hash`，存储前保留已有文件冲突检查。它们的旧参数、请求画像、输入配额、计算时间和块大小规则不因本次 placement 修改而变动。

## 旧结果的适用范围

已保存的 manifests、原始运行结果、CSV、PNG、PDF、PPT 和审计记录保持原始字节。此前 SSU3 的90.68%等利用率来自旧条带输入，不能作为 Ring hash 的实测结果。本次是代码迁移与回归验证，没有批量重跑这些正式实验。

`load_manifest`、显式 `--manifest` 和历史分析脚本仍可读取已有落盘清单；它们不生成被删除的 placement，也不会把旧清单重新 hash。`audit_*` 中验证旧取模位置的断言，以及 `miss64_math.py` 中的旧条带数学估计，属于历史证据的核验口径，未改成 Ring hash 结论。

严格历史回放会核验原源码 SHA。修改输入生成器后，旧 SHA 检查可能拒绝当前工作树，这是预期行为；应恢复该次运行记录的源码版本或源码归档，不能修改旧 SHA 来绕过。新实验应保存到新的目录，随后重新测量逐盘负载、NPU利用率和SLO。
