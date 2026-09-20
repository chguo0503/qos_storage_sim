# KV 命中后，为什么还要算最后一个 token？

[阅读六页 PDF：vLLM 通俗图解](KV命中后为什么还要算最后一个token_vLLM图解.pdf)

内容从“今天｜天气｜很 → 好”的示例开始，区分最后一个输入 token 和第一个输出 token，再解释 KV、最终隐藏状态、logits，以及尾部 token 如何逐层使用缓存。

第4页按普通全注意力、本地前缀缓存、有效对齐粒度16的配置，说明输入35、32、33个token时，为什么分别可能重算3、16、1个。第5页对应当前 ToolAgent 转换文件中的请求，区分源码规则推导、合成时间和实际设备测量。

源码核对日期：2026-09-19；固定官方 vLLM 提交 `a8d1aa9c99b8698a2a78b611b7a10c30e6b3995b`。PDF中的链接可点击。没有运行真实 vLLM 性能测量，也没有改动 trace、转换规则或仿真器。

- [来源](sources.json)
- [生成脚本](build_guide.py)
- [数值与文字检查](validation.json)
- [页面边界检查](layout_checks.json)

重新生成：在项目根目录执行 `python docs/kv_cache_last_token/build_guide.py`。使用 ReportLab 和本机已有的微软雅黑字体；PDF已嵌入字体，阅读时不需要安装同一字体。
