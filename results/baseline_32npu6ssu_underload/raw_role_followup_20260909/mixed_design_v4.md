# v4：176K原始全局人口重绑，Long从t=0同步开始

[mixed_design_v4.py](mixed_design_v4.py)冻结SHA-256为`fbd970295e39e8058282238c71ff8e1fbdfe0a92d96d75af38bbbc5358f445f2`。API：`build_queues(source_requests, source_metadata, seed, mode)`，返回新NPU→原请求对象列表及描述；mode为`random`或`ordered`。调用者显式更改NPU与位置ID，并记录源ID/NPU。模块不修改原对象、C/V、arrival=0或物理placement。

来源为真实`raw176_three_l20_fixed_seed7`：1212条请求，176K/1024 Long420条，32/48/64K、NQL1024 Short各264条。**全局人口不变，但逐卡人口重新分配**，因此排序的配对对照必须是v4新绑定的random，不能只拿旧raw176 mixed/random作为单因素对照。

| 新卡号 | 每卡Long数 | 每种Short数 | Ordered |
|---|---:|---:|---|
| 0–16 | 15 | 3 | 全部L→三种Short轮转 |
| 17–19 | 13 | 4 | 全部L→三种Short轮转 |
| 20–25 | 11 | 17 | 下述共同结构 |
| 26–28 | 10 | 17 | 下述共同结构 |
| 29–31 | 10 | 16 | 下述共同结构 |

后12卡依次为：**独立洗牌9条S2与8条S3 → 连续16条S1 → 全部Long → 剩余Short轮转**。没有删除请求。画像S1/S2/S3对应32/48/64K，三个Short实际分类都是SL，Long为LL。random模式对每张新卡的整个完整列表独立洗牌，不重复固定deck。绑定的随机流与排序的随机流分离，两模式每卡原对象集合严格相同。

17张主Long卡前段纯计算3769.970286602ms，3张Guard前段3267.307581722ms。初次读和后续停顿只会推迟这些段结束，不能将纯C当成实际交接时刻。后12卡混合S2/S3段纯C1523.854433718ms，加16条S1后总前缀纯C2452.780075840ms。每卡完整纯C最少4221.699863275ms；这是充足人口证据，实际暖窗是否全active仍由日志核验。

目的在于消除v2/v3的共同Short前缀对Long入场相位的扰动：20张Long卡直接在t=0读第一层，且共同Long C/V在后续连续请求中保持。后12卡先处理计算较长的两种Short，再将更短S1集中到主要观察窗。相位是否维持仍须比较逐层release/compute spread，不能由排法直接认定。

已有176K固定Baseline中，暖窗内完整处理的S1/S2/S3均值为84.492211333、103.520588711、122.758404965ms（样本76/65/79）。将这些**旧输入**的均值当作候选尺度，后12卡混合段约1913.752538ms，全部前缀约3265.627919ms。它接近Guard纯C3267.307582ms，而Guard还有初次读成本，所以存在短暂32Long重叠风险，不能宣称Guard必已离开。代理来源`runs/raw176_three_l20_fixed_seed7/baseline/raw176_three_l20_fixed_seed7_9a41d5b9643b_baseline_104671470b.json.gz`，SHA-256 `83fbf0a74c9b09868c45f919e793ab4a00502891e556ab210501727589b2cb0d`。代理未用于强行控制时间。

复算器[mixed_design_v4_check.py](mixed_design_v4_check.py)已对真实seed7的32卡完成random/ordered配额、同对象、全局完整性、输入枚举反转后的确定性检查，全部通过。新的静态逐盘充分证书最热仍为40.0057032375GiB/s，因此**未获得任意同时组合都欠载的证明**。这不是输入完整性错误；两模式、两策略的实际全程需求都要另外扫描。K≤31时仍有39.872203868GiB/s的通用上界；K=32时实际放置决定是否超限。

还必须逐项报告每卡暖窗Long/Short正计算时长、全窗活跃、实际角色并发、窗口设备U及暖窗admission队列的完整处理SLO。若random某卡暖窗没有两角色，或ordered没有达到目标，都保留报告，不筛去该seed。新分卡造成不同卡总纯C不均衡，完整人口makespan与整机U可能受尾部影响，不能用暖窗U代表全程。复算证据：[mixed_design_v4_checks.json](mixed_design_v4_checks.json)。本文件描述生成与数学检查，没有运行新仿真。
