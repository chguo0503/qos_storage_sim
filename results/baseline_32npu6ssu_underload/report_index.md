**NPU利用率与TTFT SLO报告索引**

硬件为32 NPU、6 SSU，主统计窗口为[2,4)秒。下列报告来自已完成实验；Once均指Once per layer。

- [混合请求Random/Ordered：seed7图对应结果](figures/separate/utilization_ttft_slo_seed7.md)
- [固定长卡/短卡：seed7图对应结果](role_separated_5seeds/figures/separate/utilization_ttft_slo_seed7.md)
- [data随机抽样：利用率及TTFT SLO，含seed7和五seed均值](raw_data_random_5seeds/utilization_ttft_slo.md)
- [data随机抽样：四个类别各自的TTFT SLO](raw_data_random_5seeds/category_ttft_slo.md)
- [data随机抽样：整机平均NPU利用率](raw_data_random_5seeds/overall_npu_utilization.md)

TTFT SLO采用接纳后处理时间代理，不包括接纳前排队；具体公式、人口和限制见各报告。完整data抽样与预筛欠载子集须分别解读。
