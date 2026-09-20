# 复现 FIFO 欠载探索

在解压后的目录运行，依赖见 `requirements.txt`（Python 3.11，numpy、matplotlib）。结果保存在 `results/fifo_underload_exploration_20260914/`。以下使用新的 stage 名称，保留随包提供的原始结果；再次运行时请更换 stage 名称，脚本拒绝覆盖已有 result。

## 固定长短角色的配对验证

```bash
python run_fifo_proof.py --case 0 --policy fifo --horizon-ms 4500 --window 2000 4000 --stage rerun_formal
python run_fifo_proof.py --case 0 --policy short_first --horizon-ms 4500 --window 2000 4000 --stage rerun_formal
python run_fifo_proof.py --case 2 --policy fifo --horizon-ms 4500 --window 2000 4000 --stage rerun_formal
python run_fifo_proof.py --case 2 --policy short_first --horizon-ms 4500 --window 2000 4000 --stage rerun_formal
```

case 0：3 SSU，19 张卡承载 200K/2048，13 张卡承载 32K/2048。
case 2：4 SSU，20 张卡承载 200K/2048，12 张卡承载 32K/1024。
其中“长度/NQL”分别是总输入长度和本次新计算 token 数，K=1024 tokens。原始 data 行直接提供单层计算时间和读取量。

## 每卡长度/NQL 组合不重复的小扰动验证

```bash
python explore_fifo_tiny_jitter.py --policy fifo --horizon-ms 4500 --window 2000 4000 --stage rerun_tiny
python explore_fifo_tiny_jitter.py --policy short_first --horizon-ms 4500 --window 2000 4000 --stage rerun_tiny
```

此组仍固定 19 张长卡、13 张短卡，在各卡内无放回随机取长度/NQL；不是所有卡随机混排长短类。

## 所有卡独立随机混排的长窗口验证

```bash
python explore_fifo_mixed20.py --case 4 --horizon-ms 4500 --window 2000 4000 --stage rerun_mixed
python explore_fifo_mixed20.py --case 9 --horizon-ms 4500 --window 2000 4000 --stage rerun_mixed
```

case 4：4 SSU，200K/2048 : 24K/1024 = 1:4；case 9：3 SSU，200K/2048 : 20K/1536 = 1:8。20K/24K 的计算时间使用原始 32K、48K 行线性外推，NQL=1536 在相邻 NQL 行之间插值；属于构造输入。

## 筛选与汇总

```bash
for i in {0..11}; do python explore_fifo_underload.py --case "$i" --stage rerun_screen; done
for i in {0..9}; do python explore_fifo_mixed20.py --case "$i" --stage rerun_mixed20_screen; done
```

广幅随机且每卡组合不重复的两组：

```bash
python explore_fifo_variants.py --case 1 --long-range 176 200 --short-range 20 35 --stage rerun_variant
python explore_fifo_variants.py --case 3 --long-range 176 200 --short-range 20 35 --stage rerun_variant
```

原始调用参数保存在各组 metadata.json。

`--horizon-ms` 用来生成足够覆盖时间窗口的请求数量，不是强行截断仿真。所有请求在 t=0 到达，各 NPU 串行接纳，仿真运行至全部请求完成。所有正式组固定 seed=7、32 NPUs、8 层、batch=1 和 ring hash 放置。

`python summarize_fifo_exploration.py` 重新核算随包附带的正式结果并生成报告及 CSV/JSON 汇总。`python analyze_fifo_trace.py` 重建固定 3-SSU 反例的关键 IO 时序图和不重叠等待时间账。完整原始结果包含 manifest、metadata、metrics、result，固定角色配对组还包含逐 IO trace。

原模拟器文件保持 GitHub commit 75e10b8a84d4054921cd3149af40507444efb3fb 不变，原始 Git blob SHA 列于 SOURCE_PROVENANCE.json；新实验说明见 EXPLORATION_PROVENANCE.json。
