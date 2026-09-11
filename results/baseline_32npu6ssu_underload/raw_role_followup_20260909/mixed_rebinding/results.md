**新绑定下的同每卡人口随机／重排比较**

U uses actual clipped compute/[32*2000ms]. Warm SLO uses admissions in [2,4)s, completion-admission<=1.5*8C followed to completion. Role counts are admitted active requests, including their stalls; idle is retained. Nominal capacity is scanned over the entire run; extra next-request L0 is not added to current-profile V/C. Scientific failures remain in all rows and completed-group means.

完成 34/34；所有运行状态、科学条件失败均保留。原固定人口只改一次绑定，之后 random/ordered 每卡身份相同。

|组|seed|顺序|策略|状态|设备U%|暖接纳SLO%|平均Long/Short|暖混合卡数|两角色各≥100ms卡数|全程盘峰GiB/s|每卡混合且欠载|
|---|---:|---|---|---|---:|---:|---|---:|---:|---:|---|
|raw176_extendedhot|101|ordered|baseline|complete|89.4803|97.0402|18.366/13.634|32|32|38.201875|True|
|raw176_extendedhot|101|ordered|once|complete|99.2396|100.0000|24.516/7.484|32|32|39.930906|True|
|raw176_extendedhot|19|ordered|baseline|complete|89.4873|97.0464|18.349/13.651|32|32|38.238878|True|
|raw176_extendedhot|19|ordered|once|complete|99.2617|100.0000|24.536/7.464|32|32|39.946934|True|
|raw176_extendedhot|43|ordered|baseline|complete|89.5718|96.0084|18.353/13.647|32|32|38.313281|True|
|raw176_extendedhot|43|ordered|once|complete|99.3305|100.0000|24.538/7.462|32|32|39.930906|True|
|raw176_extendedhot|67|ordered|baseline|complete|89.5182|96.8487|18.331/13.669|32|32|38.297253|True|
|raw176_extendedhot|67|ordered|once|complete|99.2869|100.0000|24.480/7.520|32|32|39.936249|True|
|raw176_extendedhot|7|ordered|baseline|complete|89.5726|96.4361|18.329/13.671|32|32|38.281225|True|
|raw176_extendedhot|7|ordered|once|complete|99.2081|100.0000|24.494/7.506|32|32|39.925564|True|
|raw176_extendedhot|101|random|baseline|complete|99.9730|100.0000|20.312/11.688|32|32|38.956792|True|
|raw176_extendedhot|101|random|once|complete|99.9499|100.0000|20.363/11.637|32|32|38.820178|True|
|raw176_extendedhot|19|random|baseline|complete|99.9919|100.0000|20.574/11.426|32|30|39.343877|True|
|raw176_extendedhot|19|random|once|complete|99.9262|100.0000|20.523/11.477|32|30|39.338534|True|
|raw176_extendedhot|43|random|baseline|complete|99.9842|100.0000|21.347/10.653|32|31|38.895690|True|
|raw176_extendedhot|43|random|once|complete|99.8865|100.0000|21.365/10.635|32|31|39.386223|True|
|raw176_extendedhot|67|random|baseline|complete|99.9862|100.0000|20.845/11.155|32|30|38.972173|True|
|raw176_extendedhot|67|random|once|complete|99.9505|100.0000|20.872/11.128|32|30|39.028334|True|
|raw176_extendedhot|7|random|baseline|complete|99.9825|100.0000|21.268/10.732|32|32|39.174097|True|
|raw176_extendedhot|7|random|once|complete|99.9361|100.0000|21.302/10.698|32|32|39.199692|True|
|raw176_rebinding|7|ordered|baseline|complete|91.6171|97.9021|21.222/10.778|32|32|39.535751|True|
|raw176_rebinding|7|ordered|once|complete|99.2162|100.0000|26.556/5.444|32|32|39.952277|True|
|raw176_rebinding|7|random|baseline|complete|99.9886|100.0000|21.442/10.558|32|31|39.407594|True|
|raw176_rebinding|7|random|once|complete|99.8828|100.0000|21.508/10.492|32|31|39.535751|True|
|raw200_diversehot|7|ordered|baseline|complete|91.2971|92.0993|18.018/13.982|32|32|39.814817|True|
|raw200_diversehot|7|ordered|once|complete|99.0626|100.0000|23.995/8.005|32|32|39.829024|True|
|raw200_earlyguard|7|ordered|baseline|complete|90.4232|87.0330|18.560/13.440|32|32|39.819553|True|
|raw200_earlyguard|7|ordered|once|complete|99.1191|100.0000|24.774/7.226|32|32|39.829024|True|
|raw200_earlyguard|7|random|baseline|complete|99.9004|100.0000|21.765/10.235|32|32|39.646394|True|
|raw200_earlyguard|7|random|once|complete|99.8385|100.0000|21.738/10.262|32|32|39.669523|True|
|raw200_rebinding|7|ordered|baseline|complete|88.8001|74.8858|18.697/13.303|32|32|39.814817|True|
|raw200_rebinding|7|ordered|once|complete|98.8224|100.0000|25.738/6.262|32|32|40.243858|False|
|raw200_rebinding|7|random|baseline|complete|99.9459|100.0000|21.694/10.306|32|32|39.623508|True|
|raw200_rebinding|7|random|once|complete|99.8380|100.0000|21.803/10.197|32|32|39.565740|True|

暖窗混合要求每卡两角色均有正计算；输入含两类不等于实际暖窗混合。100ms阈值另列，不偷换正计算定义。平均20/12是描述目标，两策略速度不同会产生不同角色数量。名义V/C不是SSD吞吐或无突发保证。暖接纳SLO人口随顺序/策略变化，完整人口指标保留在analysis_cache。

随机对照复用：raw200_diversehot_random_seed7 → raw200_earlyguard_random_seed7/baseline (complete); raw200_earlyguard_random_seed7/once (complete)。完整输入指纹相同且逐卡科学身份相同；不计为新增仿真或独立种子。