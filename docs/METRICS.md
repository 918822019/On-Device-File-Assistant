# 复盘指标口径（V1）

> README「产品目标与复盘指标」的落地文档。V1 复盘只看一件事：**用户再次回来**。
> 采集代码：`src/edge_cloud_agent/analytics/`；只读端点：`GET /v1/metrics`；
> Web UI：「📈 复盘」Tab。

## 数据流

```
路由埋点(record_safe 失败保护, 绝不影响业务响应)
   └─► MetricsService.record_* ─► data/metrics_events.jsonl（追加式 JSONL, 内存+落盘）
GET /v1/metrics ─► MetricsService.compute()（纯函数, 全量重算, 事件少时无性能问题）
```

事件类型（`event` 字段）：

| 事件 | 触发点 | payload 关键字段 |
|---|---|---|
| `expense.collect` | POST /v1/expense/collect 成功 | claim_id, material_id, needs_follow_up, missing_required_types |
| `expense.search` | POST /v1/expense/search 成功 | claim_id（可空） |
| `expense.export` | POST /v1/expense/export 成功 | claim_id, material_count |
| `expense.correct` | POST /v1/expense/correct **且实际改动字段** | claim_id, material_id, corrected_fields |
| `filesearch.state` | search / clarify 每轮返回 | session_id, state, round(search\|clarify) |
| `filesearch.action` | execute 正常返回（HTTPException 不计） | session_id, action, status |

## 指标定义（分子 / 分母 / 边界）

### 报销

| 指标 | 分子 | 分母 | 窗口与边界 |
|---|---|---|---|
| **14 天回访率** `revisit_rate` | 首 collect 后窗口内出现过带同 claim_id 的 search/export 的 claim 数 | 有过 collect 的 claim 总数 | 窗口 = `METRICS_REVISIT_WINDOW_DAYS`（默认 14）。**先后关系按事件顺序判定**（ts 仅秒级精度，同秒 collect→search 也算回访）；无 claim_id 的 search/export 不计入；collect 之前的 search 不算 |
| **1 小时补齐率** `followup_completion_rate` | 首次缺件后 ≤1h 内再有 collect 使该 claim 的 missing_required_types 清空的 claim 数 | 出现过 needs_follow_up=true 的 claim 数 | 窗口 = `METRICS_FOLLOWUP_WINDOW_HOURS`（默认 1）。补齐判定用 collect 响应回显的 claim 级 missing 列表 |
| **字段纠正次数** `corrections_total` | /v1/expense/correct 实际改动 ≥1 个字段的调用数（传值与现值一致不计数） | —（绝对值，越少越好） | `corrected_materials` 为涉及的去重材料件数 |

### 文件搜索

| 指标 | 分子 | 分母 |
|---|---|---|
| **追问收敛率** `clarify_resolve_rate` | 首查 needs_clarification 且后续任一轮到达 resolved 的会话数 | 首查 needs_clarification 的会话数 |
| **收敛且执行动作率** `clarify_exec_rate`（README 口径） | 上述会话中又执行过动作（execute `status=ok` 且**带 session_id**）的会话数 | 同上 |

辅助计数：`sessions_total` / `resolved_direct`（首查即命中）/ `not_found`。

## 通用约定

- **比率分母为 0 时返回 `null`（UI 显示"样本不足"），不以 0.0 冒充**——假指标比没指标更糟，与本仓 `extraction_confidence` 修复同一原则。
- 存储容错：半行 JSON / 缺 event/ts 的脏行读取时跳过，不炸整库。
- 重置指标：停服务后删除 `data/metrics_events.jsonl` 即可（与业务数据完全分离）。
- 落盘失败不阻断业务：事件仍在内存态，降级为本次进程可见。

## 已知偏差（读数时注意）

1. **回访率早期偏低**：分母不剔除"首 collect 距今不足 14 天"的新 claim（它们还没走完窗口）。想要成熟队列口径可自行按 `first_collect ≤ now-14d` 过滤，V1 不做。
2. **execute 不带 session_id 的动作不计入漏斗**：`FileActionRequest.session_id` 可选，Web UI 始终携带；其他客户端（如 Android 早期版本）若不带，动作能执行但不进"收敛且执行"分子。接入新客户端时注意带上。
3. 事件流全量重算：单文件 JSONL 顺序扫描，事件量到十万级前无需优化；再往上考虑按天分片或预聚合。

## 配置

```bash
METRICS_STORE_PATH=data/metrics_events.jsonl   # 事件流位置
METRICS_REVISIT_WINDOW_DAYS=14                 # 回访窗口
METRICS_FOLLOWUP_WINDOW_HOURS=1                # 补齐窗口
```
