"""共享工具层:与业务无关、只依赖 stdlib 的跨平台基础能力。

- text_utils:  中英混合分词(两条业务线统一口径,jieba 可选降级)
- path_utils:  平台判定 / file URI 规范化与还原 / Windows 路径映射
- file_io:     编码降级读取(utf-8-sig → gb18030)+ 二进制启发式
- source_discovery: 四平台索引源目录探测(scripts/sources.py 的逻辑层)

依赖方向:本包是全包最底层,不导入 edge_cloud_agent 其他模块
(source_discovery 对 path_utils 的导入为包内依赖)。
"""
