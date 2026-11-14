"""按走访预算选择可作业窗口，并输出可信窗口数与不确定性。

标签语义：
- worked 的 work_start..work_end 是“区间内至少一天可作业”的证据，不表示
  每天都可作业，也不把区间填报日当作作业日；
- not_workable 的走访日是可作业的否定证据；
- unknown 表示已作业但日期不详，不进入逐日训练；
- unvisited 没有作业信息，不进入训练。

窗口概率来自区间删失逻辑模型，不确定性来自对系数后验的固定种子抽样。
"""

import math
from datetime import timedelta

from .features import HEAVY_RAIN_MM
from .model import DRAW_COUNT

CREDIBLE_LOWER = 0.5
MIN_LABELLED_PER_CLASS = 3
MIN_SAMPLES = 12
MIN_CROP_INTERVALS = 3
MAX_FEATURES = 3
SELECTION_GAIN = 1.0
REFERENCE_BUDGETS = (1, 2, 3, 4, 6, 8)
INTERVAL = (0.1, 0.9)


def _conditions_ready(examples):
    positives = sum(1 for example in examples if example["label"] == 1)
    negatives = len(examples) - positives
    return (positives >= MIN_LABELLED_PER_CLASS and negatives >= MIN_LABELLED_PER_CLASS
            and len(examples) >= MIN_SAMPLES)


def _varying_indices(examples):
    rows = [row for example in examples for row in example["features"]]
    if not rows:
        return []
    return [index for index in range(len(rows[0]))
            if max(row[index] for row in rows) > min(row[index] for row in rows)]


def _select_by_search(examples, indices):
    """按 BIC 逐步选择：样本少时强惩罚，没有稳定增益就退回仅截距模型。"""
    from .model import IntervalLogistic
    penalty = math.log(max(len(examples), 2))
    chosen = []
    best = 2.0 * IntervalLogistic().fit(examples, []).loss + penalty
    model = None
    while len(chosen) < MAX_FEATURES:
        candidates = []
        for index in indices:
            if index in chosen:
                continue
            candidate = IntervalLogistic().fit(examples, chosen + [index])
            score = 2.0 * candidate.loss + (len(chosen) + 2) * penalty
            candidates.append((score, index, candidate))
        if not candidates:
            break
        score, index, candidate = min(candidates, key=lambda item: (item[0], item[1]))
        if best - score < SELECTION_GAIN:
            break
        best, chosen, model = score, chosen + [index], candidate
    if model is None:
        model = IntervalLogistic().fit(examples, [])
    return chosen, model


def fit_model(examples, names):
    """训练日级可作业模型；样本不足时返回 None。"""
    if not _conditions_ready(examples):
        return None
    selected, model = _select_by_search(examples, _varying_indices(examples))
    model.feature_names = [names[index] for index in selected]
    model.training_examples = list(examples)
    model.informative = bool(selected)
    return model


def horizon_days(cutoff, horizon):
    return [cutoff + timedelta(days=offset) for offset in range(1, int(horizon) + 1)]


def _heavy_run(rows):
    longest = current = 0
    for row in rows:
        if row["rain_mm"] >= HEAVY_RAIN_MM:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _history(conditions, plot, cutoff_day):
    """历史经验边界；地块样本少时与同村或同作物地块合并。"""
    plot_rows = [row for day in conditions.days.get(plot, []) if day <= cutoff_day
                 for row in [conditions.row(plot, day)] if row]
    row = conditions.row(plot, cutoff_day)
    crop = row["crop"] if row else conditions.crop(plot)
    village = row["village_id"] if row else None
    rows = plot_rows
    pooled = len(plot_rows) < 10
    if pooled:
        rows = [sample for (other, day), sample in conditions.rows.items()
                if day <= cutoff_day and (other == plot or sample["crop"] == crop
                                          or sample["village_id"] == village)]
    rain = [sample["rain_mm"] for sample in rows]
    wetness = [sample["soil_wetness"] for sample in rows if sample["soil_wetness"] is not None]
    return {"rain_max": max(rain) if rain else 0.0,
            "wetness_max": max(wetness) if wetness else 1.0,
            "heavy_run": _heavy_run(rows),
            "days": len(rows), "plot_days": len(plot_rows), "pooled": pooled}


def _window_extremes(conditions, plot, days):
    rain = wetness = 0.0
    rows = []
    missing = 0
    for day in days:
        row = conditions.row(plot, day)
        if row is None:
            missing += 1
            continue
        rows.append(row)
        rain = max(rain, row["rain_mm"])
        if row["soil_wetness"] is not None:
            wetness = max(wetness, row["soil_wetness"])
    return {"rain_max": rain, "wetness_max": wetness,
            "heavy_run": _heavy_run(rows), "missing_days": missing}


def experience_gate(conditions, plot, cutoff_day, days, training):
    """作物支持少、地块历史少或未来一周超出历史观测尺度时停止精确建议。"""
    history = _history(conditions, plot, cutoff_day)
    window = _window_extremes(conditions, plot, days)
    crop = conditions.crop(plot)
    all_crops = sorted({row["crop"] for (other, day), row in conditions.rows.items()
                        if day <= cutoff_day})
    crop_intervals = [example for example in training
                      if example["label"] == 1 and conditions.crop(example["plot_id"]) == crop]
    reasons = []
    if len(all_crops) < 2:
        reasons.append("可用作物只有 {} 种，无法校准作物差异".format(len(all_crops)))
    if len(crop_intervals) < MIN_CROP_INTERVALS:
        reasons.append("作物 {} 的可作业区间只有 {} 段，不足以校准".format(
            crop, len(crop_intervals)))
    if history["plot_days"] < 10:
        reasons.append("该地块历史条件不足 10 天")
    if window["missing_days"]:
        reasons.append("未来一周缺少 {} 天的地块条件".format(window["missing_days"]))
    if window["heavy_run"] > history["heavy_run"]:
        reasons.append("连续强降雨 {} 天超过历史经验 {} 天".format(
            window["heavy_run"], history["heavy_run"]))
    if window["rain_max"] > history["rain_max"] + 1e-9:
        reasons.append("日降雨 {:.1f} mm 超过该地块历史上限 {:.1f} mm".format(
            window["rain_max"], history["rain_max"]))
    if window["wetness_max"] > history["wetness_max"] + 1e-9:
        reasons.append("土壤含水率 {:.2f} 超过该地块历史上限 {:.2f}".format(
            window["wetness_max"], history["wetness_max"]))
    return {"sufficient": not reasons, "reasons": reasons,
            "history": history, "window": window, "crop": crop,
            "crop_work_intervals": len(crop_intervals)}


def _candidate(model, conditions, plot, day):
    features = conditions.features(plot, day)
    if features is None:
        return None
    interval = model.bag_interval([features])
    return {"plot_id": plot, "day": str(day),
            "probability": round(interval["mean"], 4),
            "lower": round(interval["lower"], 4),
            "upper": round(interval["upper"], 4),
            "credible": interval["lower"] >= CREDIBLE_LOWER,
            "interval_level": list(INTERVAL)}


def _rank(candidates):
    return sorted(candidates, key=lambda item: (-item["lower"], -item["probability"], item["day"]))


def build_advice(model, conditions, requests, cutoff_day):
    """对每个请求给出预算内的可排期窗口；证据不足时撤回精确建议。"""
    plans = []
    training = getattr(model, "training_examples", [])
    informative = getattr(model, "informative", False)
    for request in requests:
        days = horizon_days(cutoff_day, request["horizon_days"])
        plot = request["plot_id"]
        budget = max(0, int(request["visit_budget"]))
        gate = experience_gate(conditions, plot, cutoff_day, days, training)
        entry = {"request_id": request["request_id"], "plot_id": plot,
                 "cutoff": request["cutoff"], "horizon_days": int(request["horizon_days"]),
                 "visit_budget": budget, "advice": "ranked",
                 "reasons": gate["reasons"], "experience": gate,
                 "model_informative": informative}
        if model is None:
            entry.update({"advice": "withheld", "windows": [], "credible_windows": 0,
                          "expected_workable_days": 0.0,
                          "reasons": ["训练标签不足，无法校准窗口概率"]})
            plans.append(entry)
            continue
        candidates = [item for item in (_candidate(model, conditions, plot, day) for day in days)
                      if item is not None]
        entry["daily"] = candidates
        if not gate["sufficient"]:
            entry.update({"advice": "withheld", "windows": [], "credible_windows": 0,
                          "expected_workable_days": 0.0})
            plans.append(entry)
            continue
        if not informative:
            entry.update({"advice": "baseline", "windows": [], "credible_windows": 0,
                          "expected_workable_days": 0.0,
                          "reasons": gate["reasons"] + [
                              "资料中未发现可辨识的天气—作业关系，窗口概率停留在基准率，"
                              "不据此排期"]})
            plans.append(entry)
            continue
        selected = _rank(candidates)[:budget]
        entry.update({
            "windows": selected,
            "credible_windows": sum(1 for item in selected if item["credible"]),
            "expected_workable_days": round(sum(item["probability"] for item in selected), 4),
            "posterior_draws": DRAW_COUNT,
        })
        plans.append(entry)
    return {"advice": plans}


def budget_ladder(advice, budgets=REFERENCE_BUDGETS):
    """比较不同走访预算：每个预算下能排进日程的可信窗口数与期望可作业天数。

    对每个请求模拟“把排期上限压到 B”，因此既比较不同实地走访预算下的
    产出，也保持随 B 单调不减。withheld 与 baseline 请求都不产窗口。
    """
    rows = []
    ranked = [plan for plan in advice["advice"] if plan["advice"] == "ranked"]
    for budget in budgets:
        credible = 0
        expected = 0.0
        for plan in ranked:
            take = plan["windows"][:min(budget, plan["visit_budget"])]
            credible += sum(1 for item in take if item["credible"])
            expected += sum(item["probability"] for item in take)
        rows.append({"visit_budget": budget,
                     "ranked_requests": len(ranked),
                     "credible_windows": credible,
                     "expected_workable_days": round(expected, 4),
                     "withheld_requests": sum(1 for plan in advice["advice"]
                                              if plan["advice"] == "withheld"),
                     "baseline_requests": sum(1 for plan in advice["advice"]
                                              if plan["advice"] == "baseline")})
    return rows
