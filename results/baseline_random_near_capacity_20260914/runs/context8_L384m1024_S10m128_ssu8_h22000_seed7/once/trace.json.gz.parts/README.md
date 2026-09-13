# 原始trace的无损分片

原始压缩trace超过GitHub单文件100 MiB限制，因此按原始字节切为每片最多64 MiB。没有丢弃数据，也没有重新压缩。

从项目根目录运行本研究的 `publish_large_artifact.py --restore 本目录/manifest.json`，将在上一级恢复原名文件；按分片和整体SHA256逐项验证。已存在且SHA正确的原文件不会被覆盖。

原始输入、结果、绘图证据和命令中的SHA保持不变。具体字节数、顺序与SHA见 [manifest.json](manifest.json)。
