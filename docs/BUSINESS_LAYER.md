# 业务层现状梳理

> 更新时间：2026-09-18（对应提交 `0ff913f` 业务层修复 + `b555a24` 仓库清理之后）。
> 本文只覆盖业务层（报销 / 个人文件搜索 / 共享基础设施），端侧推理与路由见
> [ARCHITECTURE.md](ARCHITECTURE.md)。

## 总览

两条业务线 + 一层共享基础设施，全部本地存储（JSONL），无外部数据库依赖：

```
                    ┌────────────────────────────────────────┐
                    │        FastAPI (main.py)               │
                    │  trace_id 中间件 / 统一异常 / 启动装配  │
                    └───────┬───────────────────┬────────────┘
             ┌──────────────┴─────┐      ┌──────┴──────────────┐
             │ 个人文件搜索 (4 API)│      │   报销闭环 (4 API)  │
             │ personal_search/   │      │   expense/          │
             └──────────────┬─────┘      └──────┬──────────────┘
                    ┌───────┴───────────────────┴─────┐
                    │ 共享: text_utils.tokenize       │
                    │      embedding_runtime(可降级)  │
                    │      JSONL store(原子写)        │
                    └─────────────────────────────────┘
```

---

## 一、个人文件搜索（主打场景）

产品形态：**「模糊口述 → 追问消歧 → 命中动作」**，disambiguation-first。

### 1.1 API 与状态机

| API | 职责 |
|---|---|
| `POST /v1/search-agent/search` | 首查，建 session，返回 `resolved / needs_clarification / not_found` + 候选 + 追问话术 + 建议动作 |
| `POST /v1/search-agent/clarify` | 追问轮：`selected_file_id` 直接确认，或 `reply` 线索收敛 |
| `POST /v1/search-agent/execute` | 命中后动作：`open / share / compare / annotate / archive` |
| `POST /v1/search-agent/rebuild-index` | 手动触发扫描（返回 scanned/imported/skipped/errors/removed） |

### 1.2 索引管道

```
三个触发入口: 后台watch线程(scan_interval) │ /search节流异步刷新(窗口内≤1次,不阻塞请求) │ rebuild-index
                    │  (_SCAN_LOCK 全局串行)
                    ▼
     扫描多根 source_dir(逗号分隔;后缀白名单+排除目录剪枝) ─► hash/size 判重【前置】─未变─► skip(不读内容/不算embedding)
                    │新增/变更
                    ▼
     构建 item: captured_at=文件mtime │ summary(110字) │ tags │ visual_hints
               │ source_app(路径弱识别: wechat/email/gallery/camera/local) │ embedding(可选)
                    ▼
     JSONL 批量落盘(persist=False + flush, 原子写) ─► FAISS 全量重建(IndexFlatIP, 768维)
                    ▼
     幽灵清理: 磁盘已删的文件移出索引(按根保护: 某根不可达时保留其记录, 防误删)
```

要点：

- **captured_at 取文件 mtime**，不是扫描时刻——时间线索检索的锚点。
- **判重前置**：未变更文件不重读内容、不重算 embedding（watch 周期成本控制的关键）。
- **/search 不再同步扫描**：请求路径只做节流异步触发，首查延迟与文件量解耦。
- **多根源目录**（跨平台）：`FILE_MEMORY_SOURCE_DIR` 逗号分隔多根，遍历用
  os.walk + `FILE_MEMORY_SCAN_EXCLUDE_DIRS` 整棵剪枝（跨 9P 扫 /mnt/c 的成本控制；
  默认清单含 macOS 卷元数据与 Windows 系统/回收站/缓存目录）；
  配置助手统一为跨平台的 `scripts/sources.py`（逻辑层 `common/source_discovery.py`，
  自动识别 Windows 原生 / WSL / macOS / Linux；旧 `wsl_sources.sh` /
  `macos_sources.sh` 已改薄包装转发）。open 动作在 WSL 与 Windows 原生下
  附带 `windows_path`（`common/path_utils.to_windows_path`），Web UI 可复制。
- **跨平台读取**：文本按 `FILE_MEMORY_TEXT_ENCODINGS`（默认 utf-8-sig → gb18030）
  降级链读取（`common/file_io.py`），Windows GBK 文件不再静默跳过；`file_uri` 在
  Windows 盘符路径下为合法 `file:///C:/...`，POSIX 与历史格式逐字一致
  （存量索引零失配）；Windows 下 OneDrive「仅在线」占位符默认跳过，
  防扫描触发静默下载。详见 [WEB_UI.md](WEB_UI.md)「WSL 文件索引层」与
  「Windows 原生运行」。

### 1.3 检索打分（`_score_item`）

```
score = [ (文本命中/总token) × 0.65      ← tokenize 切词 + 整句命中加成 1.5
        + 余弦相似度        × 0.30      ← embedding 不可用时自动归零降级
        + 线索分            × 0.25      ← 来源0.8 / 视觉0.7 / 时间0.7 / 版本0.5
        + 版本排序加分       × 0.20 ]  ÷ 权重和(1.2)，clamp [0,1]
score ≤ 0.01 → 过滤；去重 → top_k
```

- 权重全部可用 `FILE_MEMORY_FAISS_*` 环境变量调整。
- 候选池：FAISS 就绪时取 ANN `top_k × candidate_multiplier`，不足补齐；否则全量扫描。
- 时间线索（`_has_time_match`）：今天/昨天/前天精确到天；上周≤14 天；上月≤45 天；
  本周按 ISO 周；最近 N 天按数字。

### 1.4 状态判定（`_decide_state`）

| 条件 | 状态 |
|---|---|
| 0 候选 | `not_found`（附引导语） |
| 1 候选且未强制消歧 | `resolved` |
| ≥2 候选且分差 ≥ 0.35 | `resolved` |
| 其余 | `needs_clarification`（≥3 候选与 2 候选话术不同） |

`force_disambiguation=true` 强制进追问。

### 1.5 澄清收敛（clarify 链路）

```
reply ─► _filter_candidates_by_reply 三路保留:
           ① 字面: reply/token 出现在 title/summary/tags/visual_hints/source_app
           ② 时间: "上周的那张" → 按 captured_at 匹配
           ③ 来源: "微信群里的" → 中文关键词映射(微信/群里→wechat 等)再比对
       ─► _rerank_within:
           reply 重新打分(字面 + 语义 embedding)
           final = 0.7×新分 + 0.3×旧分先验, clamp [0,1]
       ─► 再走 _decide_state → 收敛 / 继续追问 / not_found(带二次引导语)
```

已验证的典型链路：`"上周群里发的聚餐照片"` → 3 候选追问 → `"聚餐"` →
2 候选（真歧义）→ `"上周的"` → resolved 到微信那张。

### 1.6 动作执行（execute）

| 动作 | 服务端行为 | 端上行为 |
|---|---|---|
| open | 返回 `file_uri` | Android 按 content/http/file 打开，失败复制 URI |
| share | 校验 `share_to`，返回 `share_payload` | 端上调起分享 |
| compare | 返回双方时间/来源/预览 + `recommended_keep`（按 captured_at，缺失按路径字典序兜底） | 展示对比 |
| annotate / archive | 写入内存 `_annotations` / `_archived` | 状态回显 |

### 1.7 会话与日志

- session 在内存 `_sessions`（读写均持 `_session_lock`），`search_id` 即 `session_id`。
- 全链路事件：`search.started → candidate_pool → scored → completed →
  clarify.started → filtered → completed → action.*`，trace_id 贯穿，
  响应头 `x-trace-id` 回传。日志字典见 [API.md](API.md)。

---

## 二、报销闭环（V1 场景）

产品形态：**收进来 → 找回来 → 拿出去**，围绕 `claim_id` 聚合。

| API | 核心逻辑 |
|---|---|
| `POST /v1/expense/collect` | 正则抽取金额/日期/商户 → 自动标题/摘要/关键词 → 落盘 → 返回 `missing_required_types` + 报销单总额 |
| `POST /v1/expense/search` | 结构化过滤（claim/doc_types/金额区间/日期区间）→ `关键词0.7 + 语义0.3` 打分；无关键词时按 `updated_at` 倒序 |
| `POST /v1/expense/export` | 按 claim 或 material_ids 生成文本 manifest（可带 raw_text） |
| `POST /v1/expense/rebuild-index` | watch_dir 扫描：一级子目录名=claim_id，文本文件自动 collect |

字段抽取规则：

- **金额**：`金额:` / `¥`/`￥` / `X元` / `共计:` 四式，取首个命中。
- **日期**：ISO（`2026-09-01`、`2026/9/1` 等）+ `X月X日`（补当年）。
- **商户**：`商户/收款方/单位/医院/门店/机构` 标签后必须跟冒号或空白
  （防止句中词误命中），惰性捕获，遇连续空白/标点/后续字段关键词
  （金额、日期、单号等）/换行/串尾截止。

其他约定：

- 关键材料类型由 `EXPENSE_REQUIRED_DOC_TYPES` 定义（默认
  `invoice,bank_transfer,receipt,approval`），collect/search 都会回显缺失项。
- watch 目录中 pdf/图片读不出文本属预期跳过（不计 error）。
- `ExpenseStore.from_dict` 容错构造：历史 JSONL 字段增减不整行丢弃。

复盘指标（README「产品目标与复盘指标」一节定义，尚未有代码采集）：14 天回访率、
1 小时补齐率、字段纠正次数。

---

## 三、共享基础设施

| 模块 | 说明 |
|---|---|
| [common/text_utils.py](../src/edge_cloud_agent/common/text_utils.py) | 中英混合切词：标点分隔 + jieba 搜索引擎模式细分（可选依赖，缺失时降级为 ≥2 字中文短语提取的旧口径），两条业务线统一 |
| embedding_runtime | `embeddinggemma-300m`；**不可用时全链路自动降级**为纯规则打分（FAISS 回退全量扫描、语义分归零），业务不中断 |
| 存储模式 | 内存 dict + JSONL 快照；个人文件侧支持批量写（`persist=False`+`flush`）；全部 tmp + `os.replace` 原子落盘 |
| [analytics/](../src/edge_cloud_agent/analytics/) | 复盘指标：路由埋点（`record_safe` 失败保护）→ 追加式 JSONL 事件流 → `GET /v1/metrics`；口径见 [METRICS.md](METRICS.md) |
| 配置 | `PersonalFileConfig` / `ExpenseConfig` / `MetricsConfig`，全环境变量驱动，权重/阈值可调 |
| 测试 | `tests/` 35 例：分词/时间线索/状态判定/重排收敛/过滤线索/字段抽取/存储往返/增量扫描生命周期 |

---

## 四、当前边界与遗留（按影响排序）

1. ~~**无中文分词**：长查询（如"上周群里发的聚餐照片"）整体是一个 token，
   文本路命中靠子串匹配常为 0，实际靠线索分支撑~~ ✅ 已修复（2026-09，M0）：
   `text_utils.tokenize` 接入 jieba 搜索引擎模式（可选依赖，缺失自动降级为
   旧纯规则口径）；单字虚词过滤；整句精确命中由打分端 +1.5 加成承接，
   不再保留切不开的整段长 token。实测同一长查询文本路 0/1 → 4/4 命中，
   头候选 0.854 直接 resolved（此前需追问轮收敛）。
2. ~~**`"v" in haystack` 版本加分过松**~~ ✅ 已修复：改为 `_VERSION_MARK_RE`
   正则（`(?<![a-z0-9])v\d+` 或"版本"字样），且仅在查询带版本意图时才加分。
3. ~~**会话/备注/归档不持久化、无过期淘汰**~~ ✅ 已修复：备注/归档落盘
   `data/file_state.json`（`FileStateStore`，原子写）；会话 TTL 30 分钟 +
   容量上限 200，惰性淘汰。
4. **「上周」=14 天窗口较宽**：今天的文件也命中"上周"；是否收紧到 ISO 周
   是产品决策。
5. ~~**来源词表与路径识别不完全对齐**~~ ✅ 已修复：引入 `_SOURCE_TEXT_ALIASES`
   显式别名表（gallery↔image、camera↔screenshot、document↔office/pdf），
   打分与过滤两侧共用同一口径。
6. ~~小项：`extraction_confidence` 恒 1.0（假指标）、export 的 `export_time`
   语义不准（取的是材料更新时间）、`datetime.utcnow()` 弃用告警~~ ✅ 全部已修复：
   `extraction_confidence` 改为金额/日期/商户三字段命中占比；`export_time` 改为
   真实导出时刻；测试中 `datetime.utcnow()` 替换为 `datetime.now(timezone.utc)`。

---

## 五、检索演进方向：LLM wiki（已决策）

> 2026-09-18 于 tiny-llm 会话确定，本节为正式记录。

**方案**：由**端侧 LLM（tiny-llm gemma4 运行时）**在索引期为文件生成并增量
维护 wiki 化知识页；查询时以 wiki 为主要检索面。与现有 embedding+FAISS
语义检索**并存**——wiki 为主、向量兜底补召回，不是替代关系。**粒度尚未定**。

### 动机（对应当前实现的三个实际弱点）

1. 无中文分词下，长查询在字面匹配路得 0 分（见 §四.1），检索质量靠时间/来源
   线索硬撑；wiki 页是 LLM 加工后的高浓缩自然语言，字面命中率天然更高。
2. embedding 是检索端侧化路上唯一的模型依赖，而 EmbeddingGemma C++ 尚未开工；
   wiki 生成复用 tiny-llm 已对齐的 gemma4 能力，端侧模型依赖可收敛为一个。
3. `raw_text` 质量低（文件名/前 120 字符），LLM 生成的 wiki 页信息密度远高于
   原始文本，同样支撑 tags/visual_hints 等结构化线索的自动填充。

### 形态草图

```
索引期: 文件入库 ─► 端侧 tiny-llm 生成/增量维护 wiki 页
                    （这是什么 / 谁相关 / 什么事件 / 何时；聚合页粒度待定）
查询期: query ─► 命中 wiki 页（字面 + 结构导航）─► 多跳到目标文件
              └► embedding 召回作为兜底补候选，进入同一套打分/消歧流程
```

现有的 `summary / tags / visual_hints / matched_clues` 字段即 wiki 化的雏形，
差别只在生成方（正则/词表 → 端侧 LLM）。

### 未决事项

- [x] **粒度**：~~每文件一页 vs 聚合页 vs 两级混合~~ → **两级（文件页+聚合页）方向暂认**
      （2026-09-18 用户认可，待索引上限一并定案后正式生效）
- [ ] **索引上限**：现 Android 侧 `MAX_INDEX_SIZE=3000` 是"一刀切"；wiki 生成
      成本使上限成为核心预算问题，候选方案为按元数据/向量/wiki 三层分预算（讨论中）
- [ ] 增量维护：文件变更时重生成整页，还是 LLM 修补既有页
- [ ] wiki 字面路与现有三路加权（0.65/0.30/0.25）的融合方式
- [ ] 生成成本：端侧 CPU 逐文件跑 E2B 级模型的 token 成本与延迟预算
- [ ] wiki 页存储位置：`files.db` 新表 vs 独立 markdown/JSONL（对应
      [edge-runtime-data-layout.md](edge-runtime-data-layout.md) 的数据布局约定）

### 依赖

wiki 层开工依赖 tiny-llm 分期 **6c–6e**（真模型导出、i4 量化、真机实测）落地，
见 [tiny-llm-gemma4-ple-design.md](tiny-llm-gemma4-ple-design.md) §6 分期表。
在此之前，现有 embedding+规则检索继续作为工作实现演进。
