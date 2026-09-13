# 清理后的可读性与可复现性审计

审计范围：`cleanup_plan.json` 的 51 项根文件、Saturday/current 两项研究、`data` 与 `docs`。这里只新增本报告及同名 JSON；未删除文件、未重跑仿真、未修改历史图或审计记录。

结论：当前保留表足以保住两项研究的运行依赖；**不需要新增根源码保留项**。历史宽范围源码校验仍依赖拟删除文件，必须保留源码快照并在隔离目录恢复。另有 3 个文档入口、共 7 个链接需要调整；原教程另有 10 张图片在本次清理前已经缺失。

## 1. 运行依赖与历史校验不是同一件事

- 静态检查 131 个保留 Python 文件，未发现导入拟删除模块；解析错误为 0。研究脚本中未发现执行性的旧结果输入目录依赖。显式动态导入仅为保留教程脚本加载标准库 `re`。
- 两项研究 86 份 command 记录共同使用 29 项 `core_source_sha256`。全部在保留表内、全部与当前源码 SHA 相符。`source_files()` 的 `coflow_*.py`、`shared_path_*.py` 动态 glob 也已纳入这 29 项。
- Saturday 的图表、逐卡均值、PDF 和公式核验会继续验证旧文件 SHA；它们引用的根源码与两项保留研究内的输入、日志、图表都仍保留。未发现依赖拟删除根文件的 Saturday 哈希映射。
- 根目录部分通用 runner 的默认输出目录仍叫旧实验目录；清理后直接按旧默认运行会重新创建这些目录。这是输出位置，不是缺失的输入依赖。复现说明应显式指定新输出目录，不必为此改冻结源码。

## 2. 历史完整校验需要的材料

必须保留 [`audit_remote_sources.tar.gz`](audit_remote_sources.tar.gz)：106 个原始冻结文件，411,350 字节。已逐字节 SHA 核验快照与各计划的关系：

| 计划 | SHA 项数 | 清理后根目录会缺 | 快照匹配 | 保留研究文件补充匹配 |
|---|---:|---:|---:|---:|
| `audit_remote_constructed_plan.json` | 114 | 54 | 106 | 8 |
| `audit_remote_context_scale_plan.json` | 121 | 54 | 106 | 15 |
| `audit_remote_once_plan.json` | 106 | 54 | 106 | 0 |
| `audit_remote_queue_plan.json` | 106 | 54 | 106 | 0 |
| `reference_load_local_plan.json` | 106 | 54 | 106 | 0 |
| `study_plan.json` | 106 | 54 | 106 | 0 |

以上全部无缺失、无 SHA 不匹配。`audit_remote_queue.py`、各 Once/constructed/context queue 与 sync 脚本，以及 `run_reference_load_local.py` 会检查宽范围计划；不能承诺它们在清理后的瘦身根目录直接继续执行。应等既有任务及同步/验收结束再清理。

复现分两种：新的同配置实验可以使用保留的 runner；历史计划的完整源码复核则将快照解压到独立目录，再补齐计划要求的当前研究文件，并逐项校验。`audit_remote_acceptance.py --source-root <独立恢复目录>` 已支持对 106 项基础计划这样验收；其他硬编码 `ROOT` 的历史 queue/sync 应在恢复后的隔离项目中使用。原输入 `.json.gz` 字节也需保留，重新压缩不保证原压缩 SHA 相同。

保留研究中 `builders/` 的旧 renderer/zoom 版本也要保留。少数图已记录当时的 builder SHA，而同名脚本随后经授权扩展；不要改旧 audit JSON 来伪装当前脚本仍是旧版本。直接查看历史图不受影响；严格重绘应选择记录中的源码版本和对应依赖。

## 3. 只需修复三个文档入口

扫描 26 个 Markdown、1,076 个本地或本仓库 main 链接。以下 7 个目标现在存在，并已确认存在于 Git 提交 `38edfa31cb5b61da02f4d198e97dc09d18356ad6`；建议将目标改成该提交的历史永久链接，不把旧数值解释成当前研究。

| 入口文件 | 位置 | 处理 |
|---|---|---|
| `docs/L1_L2_L3_NPU_图解教程.md` | 503、506、509 行，4 个链接 | 将 GitHub `main` 改为上述历史提交 |
| `docs/cleanup_20260911/README.md` | 21、60 行，2 个链接 | 改为历史提交链接，并说明这是 9 月 11 日记录 |
| `results/baseline_ab128_32_ratio12_20260912/formula_review/README.md` | 220 行，1 个链接 | 旧持续混合研究改为历史提交链接 |

完整目标与建议 URL 已写入 [JSON 清单](cleanup_readability_audit.json)。其余 Saturday 与 `docs/l1_l2_l3_beginner/README.md` 的本地链接未发现因本次清理失效。

## 4. 清理前已经存在的问题

原始 `docs/L1_L2_L3_NPU_图解教程.md` 引用的 `docs/figures/01…08、10、11` 共 10 张 PNG 当前不存在。这不是清理导致的。建议文首提示“原稿图片不完整”，并提供完整通俗教程 `docs/l1_l2_l3_beginner/README.md` / 已生成 PDF 的入口；不要凭空替换原图或改旧科学结论。

原记录中的绝对路径、SHA、删除计划、旧命令示例属于历史证据，保留原样；导航中的“最新”“当前”入口应由主任务统一更新。该审计没有验证外部 Python/字体/Chrome 环境的可安装性；PDF 重绘仍需要原依赖与中文字体。
