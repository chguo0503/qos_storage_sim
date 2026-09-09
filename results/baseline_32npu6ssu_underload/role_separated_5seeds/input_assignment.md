# 固定长短角色分卡：seed 7 的实际输入

这里导出的是已经完成仿真的四份冻结输入：11 长卡 / 21 短卡、6 长卡 / 26 短卡，各有 random 和 round_robin 两种短卡队列顺序。Baseline 与 Once 使用同一份输入，因此每种输入只导出一次。没有生成新请求，也没有重新运行仿真。

所有请求在 t=0 到达，每条请求运行 8 层。每张 NPU 一次处理一条请求，绑定后不跨卡迁移；32 张卡共享同一组 6 块 SSU，并非每类卡拥有专用磁盘。原来的混合实验同样固定 NPU 绑定，这次改变的是把长短角色分别集中到不同卡。短卡只承担 short 角色，但包含 S1、S2、S3 三种不同画像。

## 四种请求到底是什么

| 类型 | seq_len / NQL | 路由类别 | 单层 C ms | 一条请求纯计算 8C ms | 每层实际读取 MiB | 全局条数 |
|---|---|---|---:|---:|---:|---:|
| S1 | 1K / 128 | SS | 0.527908845 | 4.223270758 | 1.203125 | 6,400 |
| S2 | 1K / 256 | SS | 0.697245498 | 5.577963987 | 1.031250 | 6,400 |
| S3 | 1K / 384 | SS | 0.899454810 | 7.195638478 | 0.859375 | 6,400 |
| L | 192K / 768 | LL | 25.612343978 | 204.898751826 | 262.968750 | 256 |

全局共 19,456 条：S1/S2/S3 各 6,400 条，L 共 256 条。四份输入与原混合实验保持同一批请求身份、计算时间、读取量和到达时刻。S1–S3 是从原 data 的 32K/48K 外推到 1K 的构造画像；S3 另在 NQL256/512 间插值。L 是 192K 的 NQL512/1024 间插值得到 NQL768。它们不是从原 data 原封不动抽出的四行。

## 卡号与配额

| 方案 | NPU 卡号（从 0 开始） | 每卡 S1 / S2 / S3 / L | 每卡总条数 |
|---|---|---|---:|
| 11 长 / 21 短 | 0–2 | 0 / 0 / 0 / 24 | 24 |
| 11 长 / 21 短 | 3–10 | 0 / 0 / 0 / 23 | 23 |
| 11 长 / 21 短 | 11–26 | 305 / 305 / 305 / 0 | 915 |
| 11 长 / 21 短 | 27–31 | 304 / 304 / 304 / 0 | 912 |
| 6 长 / 26 短 | 0–3 | 0 / 0 / 0 / 43 | 43 |
| 6 长 / 26 短 | 4–5 | 0 / 0 / 0 / 42 | 42 |
| 6 长 / 26 短 | 6–9 | 247 / 247 / 247 / 0 | 741 |
| 6 长 / 26 短 | 10–31 | 246 / 246 / 246 / 0 | 738 |

配额来源是按画像分别分发：先把该画像的原请求 ID 排序，再独立打乱身份，随后从该角色最小卡号开始逐卡轮转，直到发完。例如 6,400=21×304+16，所以 21 张短卡中前 16 张每种短画像多一条；三个画像都从同一张短卡重新开始分发，所以余数落在相同卡上。256=11×23+3；6 卡方案则有 256=6×42+4、6,400=26×246+4。

分配身份所用 Python RNG 种子为 `seed*1000003 + 长卡数*10007 + profile_index*101 + 17`，其中 profile_index 按 S1、S2、S3、L 为 0、1、2、3。本页 seed=7。先完成分卡，之后才选择 random 或 round_robin，因此两种顺序不会改变某张新卡分到哪些请求。

random：先按原请求 ID 排好该短卡完整列表，再用 `Random(seed*1000003 + npu_id*100003 + 71923).shuffle(...)` 整体打乱一次，不是反复复制一个小段。round_robin：把该卡 S1/S2/S3 各自按原 ID 排序，依次从三个队列各取一条；本批每卡三类配额相等，因此类型序列完整重复 S1、S2、S3。两种模式下，长卡都按原 ID 排序：全是同一种 L，连身份顺序也相同。规则来源：[run_role_separated.py](run_role_separated.py#L57)。

## 每卡队列与纯计算总时长

下表是输入队首 16 条类型，**不是暖窗 2–4 秒内的前 16 条请求**。纯计算总时长是该卡完整队列所有请求的 8C 之和，不含 I/O 等待，不能当作实际完成时间。完整队列见文末 CSV。

### 11 长卡 / 21 短卡

| NPU | 角色 | S1 / S2 / S3 / L | 完整队列纯 C ms | random 队首16条 | round_robin 队首16条 |
|---:|---|---|---:|---|---|
| 0 | 长 | 0 / 0 / 0 / 24 | 4917.570043812 | L L L L L L L L L L L L L L L L | L L L L L L L L L L L L L L L L |
| 1 | 长 | 0 / 0 / 0 / 24 | 4917.570043812 | L L L L L L L L L L L L L L L L | L L L L L L L L L L L L L L L L |
| 2 | 长 | 0 / 0 / 0 / 24 | 4917.570043812 | L L L L L L L L L L L L L L L L | L L L L L L L L L L L L L L L L |
| 3 | 长 | 0 / 0 / 0 / 23 | 4712.671291987 | L L L L L L L L L L L L L L L L | L L L L L L L L L L L L L L L L |
| 4 | 长 | 0 / 0 / 0 / 23 | 4712.671291987 | L L L L L L L L L L L L L L L L | L L L L L L L L L L L L L L L L |
| 5 | 长 | 0 / 0 / 0 / 23 | 4712.671291987 | L L L L L L L L L L L L L L L L | L L L L L L L L L L L L L L L L |
| 6 | 长 | 0 / 0 / 0 / 23 | 4712.671291987 | L L L L L L L L L L L L L L L L | L L L L L L L L L L L L L L L L |
| 7 | 长 | 0 / 0 / 0 / 23 | 4712.671291987 | L L L L L L L L L L L L L L L L | L L L L L L L L L L L L L L L L |
| 8 | 长 | 0 / 0 / 0 / 23 | 4712.671291987 | L L L L L L L L L L L L L L L L | L L L L L L L L L L L L L L L L |
| 9 | 长 | 0 / 0 / 0 / 23 | 4712.671291987 | L L L L L L L L L L L L L L L L | L L L L L L L L L L L L L L L L |
| 10 | 长 | 0 / 0 / 0 / 23 | 4712.671291987 | L L L L L L L L L L L L L L L L | L L L L L L L L L L L L L L L L |
| 11 | 短 | 305 / 305 / 305 / 0 | 5184.046333105 | S2 S2 S2 S2 S3 S1 S3 S1 S2 S2 S3 S3 S3 S2 S1 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 12 | 短 | 305 / 305 / 305 / 0 | 5184.046333105 | S3 S2 S1 S2 S2 S3 S2 S1 S3 S3 S2 S2 S1 S3 S1 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 13 | 短 | 305 / 305 / 305 / 0 | 5184.046333105 | S2 S2 S3 S2 S2 S2 S3 S2 S3 S3 S2 S3 S3 S3 S1 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 14 | 短 | 305 / 305 / 305 / 0 | 5184.046333105 | S1 S3 S3 S1 S2 S2 S2 S2 S1 S1 S3 S3 S3 S1 S1 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 15 | 短 | 305 / 305 / 305 / 0 | 5184.046333105 | S2 S1 S1 S3 S3 S3 S2 S1 S1 S2 S2 S3 S1 S1 S2 S2 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 16 | 短 | 305 / 305 / 305 / 0 | 5184.046333105 | S3 S2 S1 S2 S1 S1 S3 S3 S2 S3 S2 S3 S2 S1 S1 S2 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 17 | 短 | 305 / 305 / 305 / 0 | 5184.046333105 | S1 S2 S2 S1 S3 S1 S2 S3 S3 S2 S3 S2 S1 S2 S2 S2 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 18 | 短 | 305 / 305 / 305 / 0 | 5184.046333105 | S2 S2 S1 S3 S3 S2 S1 S2 S2 S3 S3 S3 S2 S3 S1 S2 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 19 | 短 | 305 / 305 / 305 / 0 | 5184.046333105 | S3 S3 S1 S2 S3 S2 S3 S2 S1 S1 S1 S2 S3 S2 S2 S3 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 20 | 短 | 305 / 305 / 305 / 0 | 5184.046333105 | S2 S3 S1 S3 S2 S1 S3 S3 S1 S3 S2 S2 S2 S1 S1 S2 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 21 | 短 | 305 / 305 / 305 / 0 | 5184.046333105 | S3 S1 S2 S1 S1 S3 S3 S2 S2 S3 S3 S3 S1 S2 S2 S2 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 22 | 短 | 305 / 305 / 305 / 0 | 5184.046333105 | S2 S3 S1 S3 S3 S3 S2 S3 S3 S3 S1 S2 S2 S1 S2 S3 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 23 | 短 | 305 / 305 / 305 / 0 | 5184.046333105 | S1 S2 S3 S1 S1 S3 S1 S3 S1 S2 S1 S1 S3 S1 S3 S2 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 24 | 短 | 305 / 305 / 305 / 0 | 5184.046333105 | S1 S2 S2 S3 S2 S3 S3 S3 S1 S3 S2 S3 S3 S2 S3 S3 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 25 | 短 | 305 / 305 / 305 / 0 | 5184.046333105 | S1 S3 S1 S2 S1 S3 S2 S2 S1 S3 S1 S1 S1 S1 S2 S2 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 26 | 短 | 305 / 305 / 305 / 0 | 5184.046333105 | S2 S1 S3 S2 S2 S1 S2 S2 S2 S3 S1 S1 S3 S3 S3 S2 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 27 | 短 | 304 / 304 / 304 / 0 | 5167.049459882 | S2 S3 S2 S2 S1 S1 S1 S3 S3 S3 S1 S2 S1 S3 S3 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 28 | 短 | 304 / 304 / 304 / 0 | 5167.049459882 | S1 S3 S3 S1 S3 S3 S1 S1 S3 S1 S3 S2 S2 S1 S2 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 29 | 短 | 304 / 304 / 304 / 0 | 5167.049459882 | S3 S1 S3 S3 S1 S3 S3 S3 S2 S1 S2 S2 S3 S1 S1 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 30 | 短 | 304 / 304 / 304 / 0 | 5167.049459882 | S1 S1 S1 S2 S2 S1 S1 S1 S1 S3 S1 S1 S3 S3 S3 S2 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 31 | 短 | 304 / 304 / 304 / 0 | 5167.049459882 | S1 S3 S2 S1 S1 S1 S3 S1 S1 S3 S2 S3 S1 S3 S1 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |

### 6 长卡 / 26 短卡

| NPU | 角色 | S1 / S2 / S3 / L | 完整队列纯 C ms | random 队首16条 | round_robin 队首16条 |
|---:|---|---|---:|---|---|
| 0 | 长 | 0 / 0 / 0 / 43 | 8810.646328497 | L L L L L L L L L L L L L L L L | L L L L L L L L L L L L L L L L |
| 1 | 长 | 0 / 0 / 0 / 43 | 8810.646328497 | L L L L L L L L L L L L L L L L | L L L L L L L L L L L L L L L L |
| 2 | 长 | 0 / 0 / 0 / 43 | 8810.646328497 | L L L L L L L L L L L L L L L L | L L L L L L L L L L L L L L L L |
| 3 | 长 | 0 / 0 / 0 / 43 | 8810.646328497 | L L L L L L L L L L L L L L L L | L L L L L L L L L L L L L L L L |
| 4 | 长 | 0 / 0 / 0 / 42 | 8605.747576671 | L L L L L L L L L L L L L L L L | L L L L L L L L L L L L L L L L |
| 5 | 长 | 0 / 0 / 0 / 42 | 8605.747576671 | L L L L L L L L L L L L L L L L | L L L L L L L L L L L L L L L L |
| 6 | 短 | 247 / 247 / 247 / 0 | 4198.227686154 | S3 S2 S2 S1 S2 S1 S3 S2 S2 S2 S2 S2 S2 S3 S1 S3 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 7 | 短 | 247 / 247 / 247 / 0 | 4198.227686154 | S2 S3 S2 S1 S1 S2 S3 S2 S2 S3 S2 S2 S1 S1 S1 S3 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 8 | 短 | 247 / 247 / 247 / 0 | 4198.227686154 | S1 S1 S1 S2 S2 S2 S1 S3 S2 S1 S3 S1 S3 S1 S2 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 9 | 短 | 247 / 247 / 247 / 0 | 4198.227686154 | S2 S3 S2 S2 S2 S2 S1 S2 S2 S3 S3 S3 S3 S1 S2 S3 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 10 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S3 S3 S2 S2 S1 S3 S1 S2 S3 S1 S1 S2 S2 S2 S2 S2 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 11 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S1 S2 S2 S3 S2 S1 S3 S1 S3 S3 S1 S1 S3 S2 S3 S3 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 12 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S1 S2 S3 S2 S2 S3 S2 S3 S1 S1 S3 S2 S1 S3 S2 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 13 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S3 S3 S3 S2 S3 S2 S3 S3 S3 S1 S1 S3 S3 S3 S2 S2 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 14 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S3 S1 S2 S3 S1 S1 S3 S2 S2 S3 S2 S2 S1 S1 S3 S3 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 15 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S3 S1 S2 S2 S1 S3 S1 S1 S1 S1 S2 S2 S1 S1 S2 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 16 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S2 S2 S3 S3 S2 S2 S1 S2 S3 S3 S1 S1 S3 S1 S1 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 17 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S1 S2 S1 S1 S3 S2 S2 S1 S1 S3 S2 S3 S3 S1 S3 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 18 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S2 S3 S2 S2 S3 S3 S3 S3 S3 S1 S3 S1 S2 S2 S3 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 19 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S2 S2 S2 S2 S2 S2 S1 S2 S1 S1 S1 S1 S3 S2 S3 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 20 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S3 S1 S1 S2 S3 S1 S3 S2 S2 S3 S3 S2 S2 S2 S2 S3 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 21 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S3 S2 S3 S3 S1 S3 S2 S1 S3 S2 S1 S3 S1 S2 S1 S2 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 22 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S3 S1 S2 S1 S3 S1 S1 S1 S2 S1 S2 S3 S2 S1 S2 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 23 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S3 S3 S1 S3 S3 S2 S3 S1 S2 S3 S2 S2 S3 S3 S3 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 24 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S3 S2 S1 S3 S3 S3 S3 S3 S1 S1 S3 S2 S3 S1 S2 S3 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 25 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S3 S3 S2 S2 S3 S1 S2 S3 S3 S2 S2 S3 S2 S1 S2 S2 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 26 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S2 S3 S3 S1 S2 S1 S1 S2 S3 S2 S2 S1 S1 S2 S2 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 27 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S1 S2 S2 S2 S3 S3 S1 S3 S3 S2 S1 S3 S3 S3 S2 S1 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 28 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S3 S1 S3 S2 S2 S3 S1 S1 S1 S2 S3 S3 S1 S3 S2 S3 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 29 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S3 S3 S2 S1 S2 S3 S3 S2 S2 S3 S3 S2 S2 S2 S2 S2 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 30 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S1 S3 S2 S3 S3 S2 S3 S3 S2 S1 S2 S3 S2 S1 S1 S3 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |
| 31 | 短 | 246 / 246 / 246 / 0 | 4181.230812931 | S2 S1 S3 S1 S3 S3 S2 S3 S3 S1 S2 S1 S1 S1 S1 S2 | S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 S2 S3 S1 |

## 共享盘与请求身份如何保留

重新分卡没有重新计算落盘位置：每条请求的每层逐块 `(SSU编号, 大小)` 列表原样保留，176 KiB 块仍在原来的盘上。原条带偏移来自 `source_original_npu_id//4`，不是新卡号；因此一张新卡可以持有来自不同原卡、条带偏移不同的请求。磁盘仍是所有 NPU 共同竞争的 6 块盘。

模拟器用 request_id 决定同到达时刻的卡内排队顺序，因此新 ID 编为 `新NPU*1000000 + 从0开始的位置`，generation 也更新为位置。`original_request_id` 保留科学上的原请求身份，`source_original_npu_id` 保留原卡号。random 与 round_robin 的模拟器输入指纹不同；本次按原身份核对，确认每张新卡的完整人口、所有计算字段、到达和逐块 placement 相同。不能把新 ID 相同直接当作同一个原请求。

四份输入所有卡的完整纯计算总量均超过 4 秒。已有运行审计确认暖窗 [2,4)s 中 32 卡都持续 active，且每卡仅计算指定角色；本导出不重新计算运行性能。每卡只有一种角色是本干预的目标，并非要求每卡只有一个短画像。

## 完整可复算文件

- [每卡配额、前16条与纯C CSV](input_assignment_by_npu.csv)：128 行，4 份输入各 32 张卡。
- [输入与配对审核 JSON](input_assignment_audit.json)：冻结文件哈希、输入指纹、原人口双射、独立重建分配/排序及 64 张卡的 random/RR 配对核对。
- [role_l11_s21_seed7_random 完整队列](input_sequences/role_l11_s21_seed7_random.csv)：19,456 行，按新卡号、队列位置排序；含原身份、原卡号、画像、C、逐盘 V 与 placement 哈希。
- [role_l11_s21_seed7_round_robin 完整队列](input_sequences/role_l11_s21_seed7_round_robin.csv)：19,456 行，按新卡号、队列位置排序；含原身份、原卡号、画像、C、逐盘 V 与 placement 哈希。
- [role_l6_s26_seed7_random 完整队列](input_sequences/role_l6_s26_seed7_random.csv)：19,456 行，按新卡号、队列位置排序；含原身份、原卡号、画像、C、逐盘 V 与 placement 哈希。
- [role_l6_s26_seed7_round_robin 完整队列](input_sequences/role_l6_s26_seed7_round_robin.csv)：19,456 行，按新卡号、队列位置排序；含原身份、原卡号、画像、C、逐盘 V 与 placement 哈希。

复算命令：`python -B export_input_assignment.py`。四份完整队列合计 77,824 行（不含表头），没有对相同请求跨配置的重复出现去重。脚本只写本说明及其导出文件，不修改冻结 plan、输入、runner 或既有分析。
