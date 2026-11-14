"""把走访资料摘要、区间标签模型与排期建议接成一个可运行的输出。"""

from .analysis import summarize
from .features import Conditions, label_examples, parse_instant
from .schedule import build_advice, budget_ladder, fit_model
from .validation import forward_validation

EVIDENCE = {
    "worked": "作业日期区间只表示区间内至少一天可作业，逐日标签由区间删失模型学习",
    "not_workable": "走访日无法作业，作为单日否定证据",
    "unknown": "已作业但日期不详，不进入逐日训练",
    "unvisited": "未走访，没有作业信息，不进入训练",
}


def training_cutoffs(tables):
    cutoffs = []
    for request in tables["requests"]:
        cutoff = parse_instant(request["cutoff"]).date()
        if cutoff not in cutoffs:
            cutoffs.append(cutoff)
    return sorted(cutoffs)


def plan(tables, validation=None):
    """生成未来一周窗口概率、不确定性、经验边界与预算阶梯。"""
    summary = summarize(tables)
    models = {}
    audits = {}
    for cutoff in training_cutoffs(tables):
        conditions = Conditions(tables, cutoff)
        examples, audit = label_examples(tables, conditions, cutoff)
        models[cutoff] = (conditions, fit_model(examples, conditions.feature_names))
        audits[str(cutoff)] = audit
    advice = []
    for cutoff, (conditions, model) in sorted(models.items()):
        requests = [request for request in tables["requests"]
                    if parse_instant(request["cutoff"]).date() == cutoff]
        advice.extend(build_advice(model, conditions, requests, cutoff)["advice"])
    if validation is None:
        validation = forward_validation(tables)
    ladder = budget_ladder({"advice": advice})
    return {
        "summary": {key: summary[key] for key in ("status_counts", "worked_intervals",
                                                   "condition_days")},
        "label_evidence": {"audit": audits, "meaning": EVIDENCE},
        "advice": advice,
        "budget_ladder": ladder,
        "validation": validation,
        "scope": "区间删失窗口概率用于走访排期参考；unknown 与 unvisited 不被当作作业日或"
                 "不可作业日；没有可辨识天气—作业关系或超出历史经验时，不输出精确建议。"
                 "验收看不同走访预算下的可信窗口产出，不看总体准确率。",
    }
