# Mixed policy 独立配对审核

新增8作业：complete=8，pending=0；另列主实验6份固定绑定参照。

所有比较使用相同 frozen input fingerprint、完整请求ID cohort、原始C与逐块SSD placement；S1/S2/S3 pipeline额外改变选卡，S3fixed保留原选卡。窗口按实际执行NPU独立积分，cohort按原manifest画像分组。逐项核心源码hash与提交种子一致，不能把pipeline收益或损失说成仅IO排序效果。

Assigned mean rho是按实际分卡后完整画像配额重新求和，仍只是平均容量必要条件；不是瞬时截止期或局部窗口的容量保证。

|输入|策略/模式|状态|U1|U2|all-active|makespan ms|短组U|短组SLO|重分卡请求数|
|---|---|---|---:|---:|---|---:|---:|---:|---:|
|aligned_short072_seed7|baseline|complete|80.292%|82.724%|True|5489.518|78.399%|82.175%|0|
|aligned_short072_seed7|once|complete|90.677%|92.904%|True|4891.038|98.435%|99.435%|0|
|aligned_short072_seed7|new_once|complete|91.620%|93.804%|True|4785.350|99.993%|100.000%|0|
|aligned_short072_seed7|strategy1_pipeline|complete|93.478%|94.424%|True|4817.828|99.927%|100.000%|18029|
|aligned_short072_seed7|strategy2_pipeline|complete|92.851%|93.748%|True|4850.463|99.856%|100.000%|18029|
|aligned_short072_seed7|strategy3_pipeline|complete|93.977%|93.620%|True|4831.173|99.885%|100.000%|18029|
|aligned_short072_seed7|strategy3_fixed|complete|90.171%|93.450%|True|4839.875|99.779%|100.000%|0|
|raw_size_varied_rho097_seed7|baseline|complete|96.235%|96.661%|True|6520.496|88.159%|81.696%|0|
|raw_size_varied_rho097_seed7|once|complete|96.365%|97.028%|True|6520.297|88.740%|90.402%|0|
|raw_size_varied_rho097_seed7|new_once|complete|97.676%|97.416%|True|6489.609|90.260%|95.685%|0|
|raw_size_varied_rho097_seed7|strategy1_pipeline|complete|95.675%|96.697%|True|6740.878|92.595%|96.280%|2870|
|raw_size_varied_rho097_seed7|strategy2_pipeline|complete|95.092%|97.210%|True|6749.121|92.760%|96.875%|2870|
|raw_size_varied_rho097_seed7|strategy3_pipeline|complete|94.940%|97.262%|True|6700.928|92.953%|96.949%|2870|
|raw_size_varied_rho097_seed7|strategy3_fixed|complete|97.832%|98.129%|True|6465.262|91.666%|94.382%|0|

## 相对各固定参照的负例

完整保留：任一窗口U较低或full makespan更长。正Δmakespan表示更慢。

|输入|策略|参照|ΔU1 pp|ΔU2 pp|Δmakespan %|
|---|---|---|---:|---:|---:|
|aligned_short072_seed7|strategy1_pipeline|new_once|+1.857|+0.620|+0.679|
|aligned_short072_seed7|strategy2_pipeline|new_once|+1.231|-0.056|+1.361|
|aligned_short072_seed7|strategy3_pipeline|new_once|+2.356|-0.184|+0.958|
|aligned_short072_seed7|strategy3_fixed|once|-0.507|+0.545|-1.046|
|aligned_short072_seed7|strategy3_fixed|new_once|-1.450|-0.354|+1.139|
|raw_size_varied_rho097_seed7|strategy1_pipeline|baseline|-0.560|+0.035|+3.380|
|raw_size_varied_rho097_seed7|strategy1_pipeline|once|-0.689|-0.331|+3.383|
|raw_size_varied_rho097_seed7|strategy1_pipeline|new_once|-2.000|-0.719|+3.872|
|raw_size_varied_rho097_seed7|strategy2_pipeline|baseline|-1.143|+0.548|+3.506|
|raw_size_varied_rho097_seed7|strategy2_pipeline|once|-1.273|+0.182|+3.509|
|raw_size_varied_rho097_seed7|strategy2_pipeline|new_once|-2.583|-0.206|+3.999|
|raw_size_varied_rho097_seed7|strategy3_pipeline|baseline|-1.295|+0.601|+2.767|
|raw_size_varied_rho097_seed7|strategy3_pipeline|once|-1.424|+0.234|+2.770|
|raw_size_varied_rho097_seed7|strategy3_pipeline|new_once|-2.735|-0.154|+3.256|

## 全画像完整cohort

|输入|策略|seqK/NQL|模拟器类别|请求数|cohort U|接纳SLO|平均stall ms|L0/warm ms|
|---|---|---|---|---:|---:|---:|---:|---:|
|aligned_short072_seed7|baseline|1/128|SS|6016|69.911%|4219/6016|1.818|0.177/1.640|
|aligned_short072_seed7|baseline|1/256|SS|6016|77.994%|5012/6016|1.574|0.197/1.377|
|aligned_short072_seed7|baseline|1/384|SS|6016|84.781%|5600/6016|1.292|0.196/1.096|
|aligned_short072_seed7|baseline|192/256|LS|576|93.404%|576/576|4.920|4.876/0.044|
|aligned_short072_seed7|once|1/128|SS|6016|97.189%|5940/6016|0.122|0.017/0.105|
|aligned_short072_seed7|once|1/256|SS|6016|98.592%|5997/6016|0.080|0.011/0.068|
|aligned_short072_seed7|once|1/384|SS|6016|99.058%|6009/6016|0.068|0.011/0.058|
|aligned_short072_seed7|once|192/256|LS|576|81.217%|500/576|16.112|6.562/9.550|
|aligned_short072_seed7|new_once|1/128|SS|6016|99.992%|6016/6016|0.000|0.000/0.000|
|aligned_short072_seed7|new_once|1/256|SS|6016|99.991%|6016/6016|0.000|0.000/0.000|
|aligned_short072_seed7|new_once|1/384|SS|6016|99.995%|6016/6016|0.000|0.000/0.000|
|aligned_short072_seed7|new_once|192/256|LS|576|82.302%|516/576|14.981|6.922/8.059|
|aligned_short072_seed7|strategy1_pipeline|1/128|SS|6016|99.876%|6016/6016|0.005|0.001/0.004|
|aligned_short072_seed7|strategy1_pipeline|1/256|SS|6016|99.939%|6016/6016|0.003|0.001/0.003|
|aligned_short072_seed7|strategy1_pipeline|1/384|SS|6016|99.948%|6016/6016|0.004|0.001/0.003|
|aligned_short072_seed7|strategy1_pipeline|192/256|LS|576|82.647%|538/576|14.628|6.899/7.729|
|aligned_short072_seed7|strategy2_pipeline|1/128|SS|6016|99.784%|6016/6016|0.009|0.002/0.008|
|aligned_short072_seed7|strategy2_pipeline|1/256|SS|6016|99.867%|6016/6016|0.007|0.001/0.006|
|aligned_short072_seed7|strategy2_pipeline|1/384|SS|6016|99.889%|6016/6016|0.008|0.002/0.006|
|aligned_short072_seed7|strategy2_pipeline|192/256|LS|576|81.420%|537/576|15.898|4.770/11.128|
|aligned_short072_seed7|strategy3_pipeline|1/128|SS|6016|99.811%|6016/6016|0.008|0.001/0.007|
|aligned_short072_seed7|strategy3_pipeline|1/256|SS|6016|99.895%|6016/6016|0.006|0.001/0.005|
|aligned_short072_seed7|strategy3_pipeline|1/384|SS|6016|99.921%|6016/6016|0.006|0.001/0.004|
|aligned_short072_seed7|strategy3_pipeline|192/256|LS|576|82.626%|536/576|14.649|5.709/8.940|
|aligned_short072_seed7|strategy3_fixed|1/128|SS|6016|99.688%|6016/6016|0.013|0.002/0.011|
|aligned_short072_seed7|strategy3_fixed|1/256|SS|6016|99.769%|6016/6016|0.013|0.002/0.011|
|aligned_short072_seed7|strategy3_fixed|1/384|SS|6016|99.841%|6016/6016|0.011|0.002/0.010|
|aligned_short072_seed7|strategy3_fixed|192/256|LS|576|81.057%|518/576|16.281|5.330/10.951|
|raw_size_varied_rho097_seed7|baseline|32/128|SS|896|57.803%|483/896|6.880|0.444/6.436|
|raw_size_varied_rho097_seed7|baseline|48/256|SS|896|86.925%|817/896|3.211|0.527/2.684|
|raw_size_varied_rho097_seed7|baseline|64/512|SL|896|98.261%|896/896|0.904|0.788/0.116|
|raw_size_varied_rho097_seed7|baseline|192/1024|LL|64|98.815%|64/64|3.272|3.272/0.000|
|raw_size_varied_rho097_seed7|baseline|192/2048|LL|192|99.513%|192/192|2.666|2.666/0.000|
|raw_size_varied_rho097_seed7|once|32/128|SS|896|76.039%|709/896|2.970|0.213/2.756|
|raw_size_varied_rho097_seed7|once|48/256|SS|896|94.505%|878/896|1.241|0.211/1.030|
|raw_size_varied_rho097_seed7|once|64/512|SL|896|89.215%|843/896|6.176|1.874/4.303|
|raw_size_varied_rho097_seed7|once|192/1024|LL|64|98.430%|64/64|4.352|4.352/0.000|
|raw_size_varied_rho097_seed7|once|192/2048|LL|192|99.357%|192/192|3.523|3.523/0.000|
|raw_size_varied_rho097_seed7|new_once|32/128|SS|896|92.446%|870/896|0.770|0.098/0.672|
|raw_size_varied_rho097_seed7|new_once|48/256|SS|896|99.488%|894/896|0.110|0.084/0.026|
|raw_size_varied_rho097_seed7|new_once|64/512|SL|896|86.528%|808/896|7.954|2.411/5.544|
|raw_size_varied_rho097_seed7|new_once|192/1024|LL|64|97.624%|64/64|6.640|6.640/0.000|
|raw_size_varied_rho097_seed7|new_once|192/2048|LL|192|99.173%|192/192|4.539|4.539/0.000|
|raw_size_varied_rho097_seed7|strategy1_pipeline|32/128|SS|896|87.531%|837/896|1.343|0.081/1.261|
|raw_size_varied_rho097_seed7|strategy1_pipeline|48/256|SS|896|96.477%|882/896|0.780|0.219/0.561|
|raw_size_varied_rho097_seed7|strategy1_pipeline|64/512|SL|896|92.031%|869/896|4.424|1.566/2.858|
|raw_size_varied_rho097_seed7|strategy1_pipeline|192/1024|LL|64|96.272%|64/64|10.564|8.864/1.700|
|raw_size_varied_rho097_seed7|strategy1_pipeline|192/2048|LL|192|98.220%|192/192|9.867|8.467/1.400|
|raw_size_varied_rho097_seed7|strategy2_pipeline|32/128|SS|896|88.458%|852/896|1.230|0.094/1.135|
|raw_size_varied_rho097_seed7|strategy2_pipeline|48/256|SS|896|96.185%|883/896|0.847|0.189/0.657|
|raw_size_varied_rho097_seed7|strategy2_pipeline|64/512|SL|896|92.215%|869/896|4.313|1.049/3.264|
|raw_size_varied_rho097_seed7|strategy2_pipeline|192/1024|LL|64|96.602%|62/64|9.595|6.398/3.197|
|raw_size_varied_rho097_seed7|strategy2_pipeline|192/2048|LL|192|98.577%|192/192|7.858|5.864/1.994|
|raw_size_varied_rho097_seed7|strategy3_pipeline|32/128|SS|896|88.964%|855/896|1.169|0.093/1.076|
|raw_size_varied_rho097_seed7|strategy3_pipeline|48/256|SS|896|95.345%|883/896|1.042|0.248/0.794|
|raw_size_varied_rho097_seed7|strategy3_pipeline|64/512|SL|896|92.747%|868/896|3.995|0.893/3.103|
|raw_size_varied_rho097_seed7|strategy3_pipeline|192/1024|LL|64|96.597%|63/64|9.611|5.872/3.739|
|raw_size_varied_rho097_seed7|strategy3_pipeline|192/2048|LL|192|98.529%|192/192|8.127|5.217/2.910|
|raw_size_varied_rho097_seed7|strategy3_fixed|32/128|SS|896|88.308%|839/896|1.248|0.111/1.137|
|raw_size_varied_rho097_seed7|strategy3_fixed|48/256|SS|896|94.248%|874/896|1.303|0.211/1.092|
|raw_size_varied_rho097_seed7|strategy3_fixed|64/512|SL|896|91.261%|824/896|4.893|1.054/3.839|
|raw_size_varied_rho097_seed7|strategy3_fixed|192/1024|LL|64|95.770%|61/64|12.050|8.887/3.162|
|raw_size_varied_rho097_seed7|strategy3_fixed|192/2048|LL|192|99.014%|192/192|5.422|4.386/1.036|
