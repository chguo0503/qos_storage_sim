# 三种短画像、无长流背景的独立控制

输入 `mixed_controls/inputs/aligned_varied_short_only_seed7.json.gz`：32 NPU、8 SSU，全部卡处理 1K/NQL128、256、384 三种短画像，每卡各 237 个请求，整条 711 请求序列独立 shuffle；seed7。共 22752 请求，所有 arrival=0，每卡 ideal compute 4028.259ms。三个 1K 画像均由原始表外推，计算时间未再缩放。全卡完整序列重放、配额、逐块 placement、fingerprint 与读量检查全部通过。

这组只有 3 种画像，作为单独控制呈现，不纳入主实验“至少 4 种画像，含短流与长流”的 12 输入、36 策略组合。

|策略|1–2s U|2–3s U|两窗32卡all-active|full makespan ms|完整短cohort compute/active|admission SLO|
|---|---:|---:|---|---:|---:|---:|
|baseline|100%|100%|是|4028.372|99.9979%|22752/22752|
|Once|100%|100%|是|4028.452|99.9969%|22752/22752|
|New Once|100%|100%|是|4028.468|99.9966%|22752/22752|

完整请求平均 IO stall 分别为 0.000119、0.000175、0.000192ms，主要是起始读入；每一种短画像 admission SLO 都是 100%。策略都完成同一个 input fingerprint `db2ac5a126b2d9cc4fab0559705963adc34fbb25db75995cace89708a39b7c29`，核心源码和提交种子一致，完整 request-ID cohorts 完全相同。Once/New Once 的完整 makespan 比 baseline 稍长约 0.08/0.10ms，此微小负例同样保留。

这个控制说明：在这组低读量短画像和当前容量下，即使每卡不断切换多种短画像、独立随机排序，baseline 也可以完全忙碌。不能仅由“短流类型多且持续到来”推出 baseline 利用率下降。

它不是相同负载下只删除长流的严格因果实验：其 mean disk rho=0.142202、link rho=0.028440，远低于主实验 aligned_short072 的 disk rho=0.935417；完整请求数、配额和序列也改变。控制不能单独区分长流读量、负载上升与瞬时队列的影响，更不能独立证明某一种 HOL 机制。

所有策略 arrival SLO 都只有 32/22752，而 admission SLO 为 22752/22752：因为所有请求 t=0 已到达，后续请求等待本卡前驱完成。此例直接展示了这两种 SLO 口径不能混用。

可复算命令：

```bash
python results/baseline_npu32_investigation/notes/aggregate_mixed_profiles.py --study results/baseline_npu32_investigation/mixed_controls --short-only-control
```

完整 CSV、含源文件 SHA256 的 JSON、逐画像 PNG/SVG 均在 `notes/mixed_control_audit/`。2026-09-07 独立只读重聚合，未运行额外仿真或修改根目录代码。
