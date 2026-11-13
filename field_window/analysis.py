"""保留区间作业日期和填报状态，不把补录日当标签。"""
from collections import Counter
from datetime import date, datetime


def visit_profile(tables):
    counts = Counter()
    intervals = []
    for row in tables["visits"]:
        if row["status"] not in {"unvisited", "not_workable", "worked", "unknown"}:
            raise ValueError("未知走访状态")
        counts[row["status"]] += 1
        if row["status"] == "worked":
            if not row["work_start"] or not row["work_end"]:
                raise ValueError("作业记录缺少可能日期区间")
            start, end = date.fromisoformat(row["work_start"]), date.fromisoformat(row["work_end"])
            if end < start:
                raise ValueError("作业日期区间反向")
            intervals.append({"visit_id": row["visit_id"], "possible_days": (end - start).days + 1})
    return {"status_counts": dict(sorted(counts.items())), "worked_intervals": intervals}


def summarize(tables):
    return {**visit_profile(tables), "condition_days": len(tables["conditions"]),
            "scope": "原始走访状态摘要，不形成可作业窗口预测"}
