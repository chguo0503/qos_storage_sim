# 同一随机输入的策略损失对照

[查看独立PNG](context384_baseline_vs_allocation_loss.png)

32NPU、8SSU、seed7，[2,20)秒。两根柱都按同一32卡×18秒分母归一化为100%。长、短计算时间均为data拟合外推，理想平均负载99.69%并不保证逐盘逐时欠载。

平均U：87.486596595% → 94.797119514%，提高7.310522919个百分点。
短请求内部等待损失减少7.001557940个百分点；首层交接等待损失减少0.308964979个百分点，二者相加与U提高量相等。
长请求内部等待、空闲均为0。首层损失包含链路固有下界，不能全部解释为FIFO。

[数值、输入一致性和来源SHA检查](checks.json) · [生成器](render_comparison.py)

[Baseline独立损失分解](../../runs/context8_L384m1024_S10m128_ssu8_h22000_seed7/baseline/figures/loss_decomposition.md)
[流量分配策略独立损失分解](../../runs/context8_L384m1024_S10m128_ssu8_h22000_seed7/once/figures/loss_decomposition.md)
