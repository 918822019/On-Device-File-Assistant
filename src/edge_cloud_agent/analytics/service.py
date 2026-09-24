"""指标记录与计算。

口径（与 README「产品目标与复盘指标」对齐，详见 docs/METRICS.md）：

报销（围绕 claim_id）：
- 14 天回访率：首次 collect 之后（按事件顺序判定先后，ts 仅秒级精度）、
  ≤14 天窗口内出现带同一 claim_id 的 search/export 事件的 claim 占比。
  无 claim_id 的 search/export 不计入。
- 1 小时补齐率：出现过 needs_follow_up=true 的 claim 中，其后（顺序）再有
  collect 使 missing_required_types 清空、且距首次缺件 collect ≤1h 的占比。
- 字段纠正次数：/v1/expense/correct 实际改动字段的事件数（越少越好）。

文件搜索（围绕 session_id）：
- 追问收敛率：首查进入 needs_clarification 的会话中，后续任一轮到达
  resolved 的占比；以及其中再执行过动作（execute status=ok 且带
  session_id）的占比（README 口径的"resolved 并执行动作"）。

比率在分母为 0 时返回 None（样本不足），不用 0.0 冒充——假指标比没指标更糟。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .storage import MetricsEvent, MetricsEventStore

# 事件名常量（埋点侧与计算侧共用，避免字符串漂移）
EV_EXPENSE_COLLECT = "expense.collect"
EV_EXPENSE_SEARCH = "expense.search"
EV_EXPENSE_EXPORT = "expense.export"
EV_EXPENSE_CORRECT = "expense.correct"
EV_SEARCH_STATE = "filesearch.state"
EV_SEARCH_ACTION = "filesearch.action"


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


class MetricsService:
    def __init__(
        self,
        store: MetricsEventStore,
        revisit_window_days: int = 14,
        followup_window_hours: int = 1,
    ) -> None:
        self.store = store
        self.revisit_window_days = revisit_window_days
        self.followup_window_hours = followup_window_hours

    # ------------------------------------------------------------------ 记录
    # 所有 record_* 由路由层调用；路由侧再包一层 try/except，双保险保证
    # 埋点永不影响业务响应。

    def record_expense_collect(
        self, claim_id: str, material_id: str, needs_follow_up: bool, missing_required_types: list[str]
    ) -> None:
        self.store.append(EV_EXPENSE_COLLECT, {
            "claim_id": claim_id,
            "material_id": material_id,
            "needs_follow_up": needs_follow_up,
            "missing_required_types": list(missing_required_types or []),
        })

    def record_expense_search(self, claim_id: str | None) -> None:
        self.store.append(EV_EXPENSE_SEARCH, {"claim_id": claim_id})

    def record_expense_export(self, claim_id: str | None, material_count: int) -> None:
        self.store.append(EV_EXPENSE_EXPORT, {"claim_id": claim_id, "material_count": material_count})

    def record_expense_correct(self, claim_id: str, material_id: str, corrected_fields: list[str]) -> None:
        self.store.append(EV_EXPENSE_CORRECT, {
            "claim_id": claim_id,
            "material_id": material_id,
            "corrected_fields": list(corrected_fields or []),
        })

    def record_search_state(self, session_id: str, state: str, round_: str) -> None:
        self.store.append(EV_SEARCH_STATE, {"session_id": session_id, "state": state, "round": round_})

    def record_search_action(self, session_id: str | None, action: str, status: str) -> None:
        self.store.append(EV_SEARCH_ACTION, {"session_id": session_id, "action": action, "status": status})

    # ------------------------------------------------------------------ 计算
    def compute(self) -> dict:
        events = self.store.list_all()
        return {
            "generated_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
            "events_total": len(events),
            "expense": self._compute_expense(events),
            "file_search": self._compute_file_search(events),
        }

    def _compute_expense(self, events: list[MetricsEvent]) -> dict:
        revisit_window = timedelta(days=self.revisit_window_days)
        followup_window = timedelta(hours=self.followup_window_hours)

        first_collect: dict[str, datetime] = {}
        first_followup: dict[str, datetime] = {}
        revisited: set[str] = set()
        followup_completed: set[str] = set()
        corrections_total = 0
        corrected_materials: set[str] = set()

        # 事件按追加顺序遍历：顺序即因果。ts 只有秒级精度，同一秒内的
        # collect→search 无法靠时间戳分先后，靠存储顺序可以。
        for ev in events:
            ts = _parse_ts(ev.ts)
            if ts is None:
                continue
            payload = ev.payload
            if ev.event == EV_EXPENSE_COLLECT:
                claim_id = payload.get("claim_id") or ""
                if not claim_id:
                    continue
                is_first = claim_id not in first_collect
                if is_first:
                    first_collect[claim_id] = ts
                # 1 小时补齐：此前登记过缺件、本次 collect 后 missing 清空
                if (
                    not is_first
                    and claim_id in first_followup
                    and claim_id not in followup_completed
                    and not (payload.get("missing_required_types") or [])
                    and ts <= first_followup[claim_id] + followup_window
                ):
                    followup_completed.add(claim_id)
                if payload.get("needs_follow_up") and claim_id not in first_followup:
                    first_followup[claim_id] = ts
            elif ev.event in (EV_EXPENSE_SEARCH, EV_EXPENSE_EXPORT):
                claim_id = payload.get("claim_id")
                t0 = first_collect.get(claim_id) if claim_id else None
                # 14 天回访：首 collect 之后（顺序保证）窗口内的 search/export
                if t0 is not None and ts <= t0 + revisit_window:
                    revisited.add(claim_id)
            elif ev.event == EV_EXPENSE_CORRECT:
                if payload.get("corrected_fields"):
                    corrections_total += 1
                    material_id = payload.get("material_id")
                    if material_id:
                        corrected_materials.add(material_id)

        claims_total = len(first_collect)
        return {
            "revisit_window_days": self.revisit_window_days,
            "claims_total": claims_total,
            "claims_revisited": len(revisited),
            "revisit_rate": _rate(len(revisited), claims_total),
            "followup_window_hours": self.followup_window_hours,
            "followup_needed_claims": len(first_followup),
            "followup_completed": len(followup_completed),
            "followup_completion_rate": _rate(len(followup_completed), len(first_followup)),
            "corrections_total": corrections_total,
            "corrected_materials": len(corrected_materials),
        }

    def _compute_file_search(self, events: list[MetricsEvent]) -> dict:
        initial_state: dict[str, str] = {}
        ever_resolved: set[str] = set()
        acted_ok: set[str] = set()

        for ev in events:
            payload = ev.payload
            if ev.event == EV_SEARCH_STATE:
                session_id = payload.get("session_id")
                state = payload.get("state")
                if not session_id or not state:
                    continue
                if payload.get("round") == "search":
                    initial_state.setdefault(session_id, state)
                if state == "resolved":
                    ever_resolved.add(session_id)
            elif ev.event == EV_SEARCH_ACTION:
                session_id = payload.get("session_id")
                # 口径：status=ok 且带 session_id 才计入"执行动作"
                if session_id and payload.get("status") == "ok":
                    acted_ok.add(session_id)

        sessions_total = len(initial_state)
        resolved_direct = sum(1 for s in initial_state.values() if s == "resolved")
        needs_clarification = [sid for sid, s in initial_state.items() if s == "needs_clarification"]
        not_found = sum(1 for s in initial_state.values() if s == "not_found")

        nc_resolved = [sid for sid in needs_clarification if sid in ever_resolved]
        nc_resolved_exec = [sid for sid in nc_resolved if sid in acted_ok]

        return {
            "sessions_total": sessions_total,
            "resolved_direct": resolved_direct,
            "needs_clarification": len(needs_clarification),
            "not_found": not_found,
            "clarify_resolved": len(nc_resolved),
            "clarify_resolved_executed": len(nc_resolved_exec),
            "clarify_resolve_rate": _rate(len(nc_resolved), len(needs_clarification)),
            "clarify_exec_rate": _rate(len(nc_resolved_exec), len(needs_clarification)),
        }
