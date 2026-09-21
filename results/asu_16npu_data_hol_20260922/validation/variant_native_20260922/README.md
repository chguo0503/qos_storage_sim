# 小型原生验证

已实际执行原生事件仿真，未修改冻结运行器或正式实验。所有验证通过。

| 验证 | 请求数 | 完成块 | 完整时间(ms) |
|---|---:|---:|---:|
| standard_asu | 4 | 9216 | 545.983656775 |
| variant_asu | 4 | 9216 | 545.983656775 |
| variant_changed_once | 4 | 9800 | 546.419327175 |
| variant_changed_once_measured | 4 | 9800 | 546.419327175 |

标准 explicit 与相同 A/B 变体输入的 ASU：完整 native_summary、所有请求/层时序、事件计数、物理输入指纹完全相同。变体 manifest 额外保存画像键和目录，所以整个 manifest 文件的 SHA 不同。

Once 使用一个不同 A：40K+17 个总 token、miss3000，计算时间来自相邻长度/miss 网格的双线性插值。被动完成回调逐请求、逐层、逐盘累计实际块数与字节，32 项全部与 manifest 一致；加观察器前后完整 native_summary 也完全相同。

逐请求每层实际计算时间、读取总量、源码前后 SHA、完整请求/块守恒和 extended 窗口统计均通过检查。小输入不到2秒，2–4/2–10/4–10秒窗口如实标 unavailable，不将它们判为通过。

配置、命令、原始结果及完成字节观察器均保存在本目录。运行 `python verify_native.py` 可只读复核原始数据；详细检查见 checks.json。
