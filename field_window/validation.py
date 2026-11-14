"""按走访预算做前向验证：只用当时可知的标签，并隔离未来补录与同场降雨。"""

from datetime import timedelta

from .features import Conditions, label_examples, parse_day
from .schedule import CREDIBLE_LOWER, fit_model

BUDGETS = (1, 2, 3, 4, 6, 8)
HORIZON = 7
GUARD_DAYS = 3
MAX_FOLDS = 4
MIN_VERIFIABLE = 2


def _event_keys(conditions, plots, horizon):
    keys = set()
    for plot in plots:
        for day in horizon:
            row = conditions.row(plot, day)
            if row is not None:
                keys.add(("plot", plot, row["process_id"]))
                keys.add(("village", row["village_id"], row["process_id"]))
    return keys


def _purge(conditions, plots, horizon):
    """训练隔离：窗口本身、前后禁带、同场降雨（同地块或同村）都不得进入训练。"""
    keys = _event_keys(conditions, plots, horizon)
    blocked = set(horizon)
    guard_start = horizon[0] - timedelta(days=GUARD_DAYS)
    guard_end = horizon[-1] + timedelta(days=GUARD_DAYS)

    def is_purged(plot, day, event):
        if day in blocked or guard_start <= day <= guard_end:
            return True
        row = conditions.row(plot, day)
        if row is None:
            return False
        return ((("plot", plot, row["process_id"]) in keys)
                or (("village", row["village_id"], row["process_id"]) in keys))

    return is_purged


def _future_labels(tables, origin, horizon):
    """窗口内的事后核对标签。

    worked 只说明区间内至少一天可作业，窗口日落在区间内记为“合理”，
    不当作逐日确认；明确 not_workable 的走访日构成否定证据。
    """
    worked, blocked = {}, {}
    for visit in tables["visits"]:
        plot = visit["plot_id"]
        if visit["status"] == "worked":
            start, end = parse_day(visit["work_start"]), parse_day(visit["work_end"])
            inside = [str(day) for day in horizon if start <= day <= end]
            if inside:
                worked.setdefault(plot, set()).update(inside)
        elif visit["status"] == "not_workable":
            day = parse_day(visit["visited_on"])
            if origin < day <= horizon[-1]:
                blocked.setdefault(plot, set()).add(str(day))
    return worked, blocked


def _auc(positives, negatives):
    if not positives or not negatives:
        return None
    wins = sum(1 for a in positives for b in negatives if a > b)
    ties = sum(1 for a in positives for b in negatives if a == b)
    return round((wins + 0.5 * ties) / (len(positives) * len(negatives)), 4)


def _fold(tables, origin, horizon_days, budgets):
    conditions = Conditions(tables, origin)
    plots = sorted({row["plot_id"] for row in tables["conditions"]})
    training, audit = label_examples(tables, conditions, origin,
                                    purge=_purge(conditions, plots, horizon_days))
    model = fit_model(training, conditions.feature_names)
    record = {"origin": str(origin), "usable": model is not None,
              "training_examples": len(training), "training_audit": audit,
              "selected_features": getattr(model, "feature_names", []),
              "horizon": [str(horizon_days[0]), str(horizon_days[-1])]}
    if model is None:
        return record
    worked, blocked = _future_labels(tables, origin, horizon_days)
    candidates = {}
    for plot in plots:
        rows = []
        for day in horizon_days:
            features = conditions.features(plot, day)
            if features is None:
                continue
            interval = model.bag_interval([features])
            rows.append({"plot_id": plot, "day": str(day),
                         "probability": interval["mean"], "lower": interval["lower"]})
        rows.sort(key=lambda item: (-item["lower"], -item["probability"], item["day"]))
        candidates[plot] = rows
    checked = []
    for plot, rows in candidates.items():
        for item in rows:
            if item["day"] in worked.get(plot, set()):
                checked.append({**item, "plausible": True, "contradicted": False})
            elif item["day"] in blocked.get(plot, set()):
                checked.append({**item, "plausible": False, "contradicted": True})
    record["checked"] = checked
    stats = {}
    for budget in budgets:
        selected = [item for plot in plots for item in candidates[plot][:budget]]
        credible = [item for item in selected if item["lower"] >= CREDIBLE_LOWER]
        stats[str(budget)] = {
            "scheduled_windows": len(selected),
            "credible_windows": len(credible),
            "plausible": sum(1 for item in selected if item["day"] in worked.get(item["plot_id"], set())),
            "contradicted": sum(1 for item in selected
                                if item["day"] in blocked.get(item["plot_id"], set())
                                and item["day"] not in worked.get(item["plot_id"], set())),
            "credible_plausible": sum(1 for item in credible
                                      if item["day"] in worked.get(item["plot_id"], set())),
            "credible_contradicted": sum(1 for item in credible
                                         if item["day"] in blocked.get(item["plot_id"], set())
                                         and item["day"] not in worked.get(item["plot_id"], set())),
        }
    record["budgets"] = stats
    return record


def _origins(tables, horizon):
    days = sorted({parse_day(visit["visited_on"]) for visit in tables["visits"]})
    usable = []
    for day in days:
        window = [day + timedelta(days=offset) for offset in range(1, horizon + 1)]
        worked, blocked = _future_labels(tables, day, window)
        count = sum(len(v) for v in worked.values()) + sum(len(v) for v in blocked.values())
        if count >= MIN_VERIFIABLE:
            usable.append(day)
    if len(usable) <= MAX_FOLDS:
        return usable
    step = (len(usable) - 1) / (MAX_FOLDS - 1)
    return [usable[round(position * step)] for position in range(MAX_FOLDS)]


def forward_validation(tables, horizon=HORIZON, budgets=BUDGETS):
    """用历史切点做前向验证，报告各走访预算下的可信窗口产出与核对结果。"""
    folds = []
    for origin in _origins(tables, horizon):
        window = [origin + timedelta(days=offset) for offset in range(1, horizon + 1)]
        folds.append(_fold(tables, origin, window, budgets))
    usable = [fold for fold in folds if fold.get("usable")]
    summary = {}
    for budget in budgets:
        rows = [fold["budgets"][str(budget)] for fold in usable]
        verified = sum(row["plausible"] + row["contradicted"] for row in rows)
        plausible = sum(row["plausible"] for row in rows)
        credible_verified = sum(row["credible_plausible"] + row["credible_contradicted"]
                                for row in rows)
        credible_plausible = sum(row["credible_plausible"] for row in rows)
        summary[str(budget)] = {
            "folds": len(rows),
            "mean_scheduled_windows": (sum(row["scheduled_windows"] for row in rows) / len(rows)
                                       if rows else None),
            "mean_credible_windows": (sum(row["credible_windows"] for row in rows) / len(rows)
                                      if rows else None),
            "mean_plausible": (sum(row["plausible"] for row in rows) / len(rows)
                               if rows else None),
            "mean_contradicted": (sum(row["contradicted"] for row in rows) / len(rows)
                                  if rows else None),
            "plausible_share_among_verified": (plausible / verified) if verified else None,
            "credible_plausible_share": (credible_plausible / credible_verified
                                         if credible_verified else None),
            "verified_windows": verified,
        }
    checked = [item for fold in usable for item in fold.get("checked", [])]
    return {"scheme": "前向时间切分：训练只用 known_at 不晚于切点的记录；"
                       "切点后的补录、跨越切点的作业区间、验证窗口及其前后 {} 天禁带、"
                       "同场降雨（同地块或同村同一事件）都从训练剔除".format(GUARD_DAYS),
            "credible_lower_bound": CREDIBLE_LOWER,
            "horizon_days": horizon, "usable_folds": len(usable),
            "verified_windows": len(checked),
            "rank_auc_among_verified": _auc([item["lower"] for item in checked if item["plausible"]],
                                            [item["lower"] for item in checked
                                             if item["contradicted"]]),
            "folds": folds, "budgets": summary}
