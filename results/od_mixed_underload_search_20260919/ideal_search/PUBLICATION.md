# 大型 JSON 的无损发布与恢复

以下三个完整搜索结果以 gzip 发布。本地原始 JSON 保留；Git 只精确忽略这三个原始路径，发布对应 `.json.gz`、SHA-256 清单与恢复脚本。没有删减候选、删除字段或重新序列化 JSON，解压后包括空白和换行在内的每一个字节都与原文件相同。

| 原始文件 | 原大小 | 发布的 gzip 大小 |
|---|---:|---:|
| `safe_valid.json` | 150.62 MiB | 5.80 MiB |
| `resonance_valid.json` | 101.21 MiB | 5.42 MiB |
| `expanded_results.json` | 48.11 MiB | 3.14 MiB |

完整的原文件与 gzip 大小、SHA-256 保存在 [large_json_archives.json](large_json_archives.json)。压缩使用 gzip level 9、时间戳 0、不写入原文件名，重复打包相同原始字节会生成相同压缩字节。

在项目根目录验证发布数据：

```bash
python results/od_mixed_underload_search_20260919/ideal_search/publish_large_json.py verify
```

验证会分别检查 gzip 自身 SHA、解压后的原始字节 SHA；若本地原始 JSON 仍存在，也检查它与发布快照一致。验证不会创建或修改结果文件。

恢复原始路径，兼容任何需要直接读取 `.json` 的工具：

```bash
python results/od_mixed_underload_search_20260919/ideal_search/publish_large_json.py restore
```

也可恢复到另一个目录：

```bash
python results/od_mixed_underload_search_20260919/ideal_search/publish_large_json.py restore --output-dir /tmp/ideal-search-restored
```

恢复前先验证全部压缩包，恢复后再次检查字节 SHA。已经存在且内容相同的文件保持原样；若已有文件内容不同，脚本报错，**不会覆盖它**。

也可直接读取压缩结果，无需恢复整个 JSON 文件：

```python
import gzip
import json

with gzip.open("safe_valid.json.gz", "rt", encoding="utf-8") as stream:
    results = json.load(stream)
```

发布前已检查项目引用：这三个原始文件仅由 `safe_search.py`、`resonance_search.py`、`expand_search.py` 写出，没有现有分析脚本直接读取它们，也没有发现历史 SHA 清单引用这些原始文件。因此没有修改搜索源码、原始结果或审计记录。若重新运行搜索生成了不同的本地 JSON，验证会明确报告它与发布快照不同；不要把新一轮结果冒充本次快照。

`pack` 子命令供维护者重现本次无损归档。它保留原始 JSON，检查压缩往返字节，并拒绝覆盖内容不同的既有压缩包或清单。
