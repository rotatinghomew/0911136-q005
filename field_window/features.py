"""从逐日地块条件构造特征，并按 known_at 与时间禁带切分走访标签。"""

from collections import Counter
from datetime import date, datetime, timedelta, timezone

HEAVY_RAIN_MM = 10.0
WET_LAGS = 4
RAIN_WINDOW = 7


def parse_day(value):
    return date.fromisoformat(value)


def parse_instant(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


STATUSES = {"unvisited", "not_workable", "worked", "unknown"}


def visit_days(visit):
    """worked 返回含两端的所有可能作业日；其他状态返回走访日。"""
    if visit["status"] == "worked":
        if not visit["work_start"] or not visit["work_end"]:
            raise ValueError("作业记录缺少可能日期区间")
        start = parse_day(visit["work_start"])
        end = parse_day(visit["work_end"])
        if end < start:
            raise ValueError("作业日期区间反向")
        return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
    return [parse_day(visit["visited_on"])]


class Conditions:
    """逐日地块条件索引；截止日之后的记录不参与特征统计。"""

    def __init__(self, tables, cutoff_day):
        self.cutoff_day = cutoff_day
        self.rows = {}
        self.days = {}
        self.crops = set()
        plot_wetness = {}
        for row in tables["conditions"]:
            plot, day = row["plot_id"], parse_day(row["day"])
            self.rows[(plot, day)] = row
            self.days.setdefault(plot, []).append(day)
            self.crops.add(row["crop"])
            if day <= cutoff_day and row["soil_wetness"] is not None:
                plot_wetness.setdefault(plot, []).append(row["soil_wetness"])
        for plot in self.days:
            self.days[plot].sort()
        self.crop_order = sorted(self.crops)
        self.reference_crop = self.crop_order[0] if self.crop_order else None
        self.plot_median = {plot: median(values) for plot, values in plot_wetness.items()}
        observed = [value for values in plot_wetness.values() for value in values]
        self.global_median = median(observed) if observed else 0.3

    @property
    def feature_names(self):
        names = ["rain_today", "wetness_today", "wetness_missing",
                 "rain_lag1", "wetness_lag1", "rain_lag2", "wetness_lag2",
                 "rain_lag3", "wetness_lag3", "rain_3day", "rain_7day",
                 "wetness_3day_max", "heavy_rain_run"]
        for crop in self.crop_order[1:]:
            names.append("crop=" + crop)
        return names

    def row(self, plot, day):
        return self.rows.get((plot, day))

    def crop(self, plot):
        for day in self.days.get(plot, []):
            return self.rows[(plot, day)]["crop"]
        return None

    def event(self, plot, day):
        row = self.rows.get((plot, day))
        return row["process_id"] if row else None

    def wetness(self, plot, day):
        row = self.rows.get((plot, day))
        if row is None or row["soil_wetness"] is None:
            fallback = self.plot_median.get(plot)
            return self.global_median if fallback is None else fallback
        return row["soil_wetness"]

    def history(self, plot, last_day):
        return [self.rows[(plot, day)] for day in self.days.get(plot, []) if day <= last_day]

    def features(self, plot, day):
        if not isinstance(day, date):
            day = parse_day(day)
        row = self.rows.get((plot, day))
        if row is None or "crop" not in row:
            return None
        rain = [0.0] * (RAIN_WINDOW + 1)
        wet = [None] * (RAIN_WINDOW + 1)
        missing = 0
        for lag in range(RAIN_WINDOW + 1):
            sample = self.rows.get((plot, day - timedelta(days=lag)))
            if sample is None:
                continue
            rain[lag] = float(sample["rain_mm"])
            wet[lag] = sample["soil_wetness"]
        if wet[0] is None:
            missing = 1
        wetness = [self.wetness(plot, day - timedelta(days=lag)) for lag in range(WET_LAGS)]
        heavy_run = 0
        for value in rain:
            if value >= HEAVY_RAIN_MM:
                heavy_run += 1
            else:
                break
        values = [rain[0], wetness[0], float(missing),
                  rain[1], wetness[1], rain[2], wetness[2], rain[3], wetness[3],
                  sum(rain[:3]), sum(rain[:RAIN_WINDOW + 1]), max(wetness[:3]),
                  float(heavy_run)]
        crop = row["crop"]
        for name in self.crop_order[1:]:
            values.append(1.0 if crop == name else 0.0)
        return values


def label_examples(tables, conditions, cutoff, purge=None):
    """返回可用的日级标签样本（正例为可能作业区间，负例为无法作业日）。

    cutoff 之后的补录、跨过截止日的区间以及会与验证目标相邻或同场降雨的
    记录都被排除，排除原因记入 audit。
    """
    audit = Counter()
    if not isinstance(cutoff, datetime):
        cutoff = datetime.combine(cutoff, datetime.max.time(), tzinfo=timezone.utc)
    cutoff_day = cutoff.date()
    not_workable_days = set()
    for visit in tables["visits"]:
        if visit["status"] != "not_workable":
            continue
        if parse_instant(visit["known_at"]) > cutoff:
            continue
        not_workable_days.add((visit["plot_id"], parse_day(visit["visited_on"])))
    examples = []
    for visit in tables["visits"]:
        status = visit["status"]
        if status not in STATUSES:
            raise ValueError("未知走访状态")
        if status == "unvisited":
            audit["unvisited"] += 1
            continue
        if status == "unknown":
            audit["worked_date_unknown"] += 1
            continue
        if parse_instant(visit["known_at"]) > cutoff:
            audit["unknown_at_cutoff"] += 1
            continue
        days = visit_days(visit)
        if any(day > cutoff_day for day in days):
            audit["overlaps_cutoff"] += 1
            continue
        plot = visit["plot_id"]
        if status == "worked" and any((plot, day) in not_workable_days for day in days):
            audit["conflicts_with_not_workable"] += 1
            continue
        if purge is not None and any(purge(plot, day, conditions.event(plot, day)) for day in days):
            audit["validation_purge"] += 1
            continue
        features = [conditions.features(plot, day) for day in days]
        if any(row is None for row in features):
            audit["missing_conditions"] += 1
            continue
        label = 1 if status == "worked" else 0
        audit["worked" if label else "not_workable"] += 1
        examples.append({"label": label, "plot_id": plot, "days": days,
                         "features": features, "visit_id": visit["visit_id"],
                         "events": {conditions.event(plot, day) for day in days}})
    return examples, dict(sorted(audit.items()))
