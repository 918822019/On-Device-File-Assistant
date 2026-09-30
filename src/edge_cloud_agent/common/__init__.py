"""共享工具层:与业务无关、只依赖 stdlib 的跨平台基础能力。

- text_utils:  中英混合分词(两条业务线统一口径,jieba 可选降级)
- path_utils:  平台判定 / file URI 规范化与还原 / Windows 路径映射
- time_utils:  全项目唯一的时间口径(naive UTC 秒精度 ISO8601)
- file_io:     编码降级读取(utf-8-sig → gb18030)+ 二进制启发式 + 流式 content_hash
               + atomic_write_text / atomic_write_lines(全项目唯一的原子落盘实现)
- vectors:     cosine_similarity(两条业务线共用的相似度口径)
- jsonl_store: JsonlSnapshotStore(JSONL 快照存储基类,PersonalFileStore /
               ExpenseStore 都建立在它上面)
- fs_scan:     目录遍历与剪枝 / size+mtime_ns 变更指纹 / 幽灵记录判定
               (两条业务线的摄取循环共用)
- watch_loop:  IngestResult + 后台循环骨架(调度、停止、失败可见)
- source_discovery: 四平台索引源目录探测(scripts/sources.py 的逻辑层)

依赖方向:本包是全包最底层,**不导入 edge_cloud_agent 其他任何模块**;包内依赖
只有 source_discovery → path_utils / file_io、jsonl_store → file_io 与
fs_scan → path_utils。这条约束
由 tests/test_architecture_layers.py 静态校验(它扫 AST,连未使用的 import 也算)。

往这里加东西的判据:如果一段逻辑在两条业务线里出现了第二份,它就该在这里。
反过来,只被一条业务线用到的东西不要放进来 —— 那会让 common/ 变成杂物间。
"""
