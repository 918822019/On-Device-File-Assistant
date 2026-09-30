# 项目结构

```text
端云结合/
├── src/edge_cloud_agent/        # Python 后端（入口 main.py）
│   ├── bootstrap.py             # 服务装配、启动扫描和停止监听
│   ├── http_handlers.py         # 请求追踪、异常处理和静态资源缓存规则
│   ├── logging_config.py        # 结构化日志配置
│   ├── common/                  # 路径、文本、时间、文件 IO 与原子写、向量度量、JSONL 存储基类、目录扫描与变更指纹、watch 循环骨架、源目录探测
│   ├── engines/                 # 端侧生成与 embedding 模型引擎
│   ├── llm/                     # 云端客户端、端云策略与生成编排
│   ├── agents/                  # 文件检索与 LLM 的业务决策编排
│   ├── personal_search/         # 文件索引、检索用例、动作、响应组装与刷新调度
│   ├── expense/                 # 报销材料业务与响应组装
│   ├── analytics/               # 事件流与复盘指标
│   └── routers/                 # HTTP 路由、接口模型与依赖读取（deps.py）
├── web/                         # FastAPI 挂载于 /web/ 的静态页面
│   ├── index.html               # 页面结构
│   ├── style.css                # 样式与移动端布局
│   ├── app.js                   # ES module 入口，负责 Tab 导航
│   └── js/
│       ├── core.js              # 请求、提示、健康检查等通用工具
│       ├── search.js            # 文件搜索、索引状态与候选动作
│       ├── chat.js              # 聊天消息与文件检索结果
│       ├── expense.js           # 报销材料页面
│       └── metrics.js           # 复盘指标页面
├── android/                     # Android 客户端
├── scripts/                     # 跨平台启动、目录配置与部署入口
├── deploy/                      # systemd 服务定义
├── tests/                       # pytest 回归用例
├── docs/                        # 使用指南、接口与端侧技术文档
├── third_party/tiny-llm/        # 独立 Git 子模块，不在主仓库直接维护
├── data/                        # 本地索引与事件流（不入库）
├── logs/                        # 运行日志（不入库）
└── models/                      # 本地模型权重（不入库）
```

## 约定

- `main.py` 只装配路由、静态页面与 lifespan；`bootstrap.py` 提供 lifespan 及服务装配、预热扫描和后台线程停止；`http_handlers.py` 注册请求追踪、异常处理与缓存规则，`logging_config.py` 配置结构化日志。
- 端侧推理加载/设备适配放 `engines/`；云端模型调用、端云策略与生成编排放 `llm/`；文件检索和对话之间的业务决策放 `agents/`。
- `personal_search/` 与 `expense/` 实现各自业务：文件动作在 `personal_search/actions.py`，索引覆盖统计在 `coverage.py`，首轮检索用例（检索 + 节流刷新 + 首轮埋点）在 `search_flow.py`，搜索结果组装在 `presentation.py`，报销响应组装在 `expense/presentation.py`。
- `routers/` 只做 HTTP 适配、日志与业务错误码映射；跨路由共用的 app.state 读取（服务就绪检查、trace_id、埋点）集中在 `routers/deps.py`，路由之间不互相调用。跨层共享工具放 `common/`；依赖方向为 `routers → agents / personal_search / expense / analytics → llm / engines → common`（`personal_search → analytics` 是首轮检索埋点）。这张图由 `tests/test_architecture_layers.py` 用 AST 静态校验，新增跨包依赖必须同时更新那里的白名单。
- 两条业务线的持久化都建立在 `common/jsonl_store.py` 的 `JsonlSnapshotStore` 上，相似度打分都用 `common/vectors.py` 的 `cosine_similarity`，原子落盘都走 `common/file_io.py` 的 `atomic_write_text` / `atomic_write_lines`。新增持久化产物时复用这三处，不要再手写 `tmp + os.replace` 或第二份余弦实现。
- 两条业务线的摄取循环共用 `common/fs_scan.py`（目录遍历与剪枝、`size`+`mtime_ns` 变更指纹、幽灵记录判定）与 `common/watch_loop.py`（`IngestResult` + 循环骨架）；平台判定统一走 `common/path_utils.is_windows_native`，不得再写第三份 `os.name == "nt"`。各业务模块的 `run_once` 只保留自己的业务编排与 `_SCAN_LOCK`（锁要保护的是 store 与向量索引的写入，而 `run_once` 有多个入口）。
- 前端使用浏览器原生 ES modules，无打包步骤；功能文件只依赖 `core.js` 或明确导入其他功能模块。`app.js` 只负责加载模块和导航。
- `scripts/run.py`、`scripts/run.sh`、`scripts/service.sh`、`scripts/sources.py` 是对外入口，路径保持稳定；`web/index.html` 通过 `/web/` 提供服务。
- `docs/` 中的端侧技术文档暂保留原路径：项目文档目录也可被文件索引扫描，移动它们会改变已入库的文件路径和外部引用。
- 本机配置写入 `.env`；`data/`、`logs/`、`models/` 为运行时数据，不应放进源码目录或提交到 Git。
