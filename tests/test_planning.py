"""验证区间标签、泄漏隔离、预算排期与停止建议。"""

import copy
import datetime as dt
import json
import random
import unittest
from pathlib import Path

from field_window.features import Conditions, label_examples, parse_instant
from field_window.planning import plan
from field_window.schedule import build_advice, budget_ladder, fit_model
from field_window.validation import _purge, forward_validation

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "fixtures" / "dataset.json").read_text())
CUTOFF = dt.datetime.fromisoformat(FIXTURE["metadata"]["cutoff"])


def fixture_tables():
    return copy.deepcopy(FIXTURE["tables"])


def rebuild(conditions_rows=None):
    """以固定种子造一份天气与作业关系清晰、作物与地块不变的资料。"""
    tables = fixture_tables()
    generator = random.Random(33)
    for row in tables["conditions"]:
        row["soil_wetness"] = round(generator.random(), 2)
        row["rain_mm"] = round(generator.random() * 30.0, 2)
        row["process_id"] = row["day"]
    lookup = {(row["plot_id"], row["day"]): row for row in tables["conditions"]}

    def workable(plot, day):
        row = lookup[(plot, day)]
        return row["soil_wetness"] < 0.5 and row["rain_mm"] < 10.0

    for visit in tables["visits"]:
        worked = workable(visit["plot_id"], visit["visited_on"])
        visit["status"] = "worked" if worked else "not_workable"
        if worked:
            start = dt.date.fromisoformat(visit["visited_on"]) - dt.timedelta(days=2)
            visit["work_start"] = str(start)
            visit["work_end"] = visit["visited_on"]
        else:
            visit["work_start"] = visit["work_end"] = None
    return tables


class LabelSemanticsTests(unittest.TestCase):
    def test_worked_interval_is_a_bag_not_daily_labels(self):
        tables = fixture_tables()
        conditions = Conditions(tables, CUTOFF.date())
        examples, _ = label_examples(tables, conditions, CUTOFF)
        worked = [example for example in examples if example["label"] == 1]
        self.assertTrue(worked)
        self.assertTrue(all(len(example["features"]) == 3 for example in worked))

    def test_unknown_and_unvisited_are_not_negative_labels(self):
        tables = fixture_tables()
        conditions = Conditions(tables, CUTOFF.date())
        examples, audit = label_examples(tables, conditions, CUTOFF)
        self.assertEqual(30, audit["worked_date_unknown"])
        self.assertEqual(30, audit["unvisited"])
        self.assertEqual({"worked", "not_workable"}, set(audit) & {"worked", "not_workable"})
        self.assertEqual(30, audit["worked"])
        self.assertEqual(30, audit["not_workable"])
        self.assertEqual(len(examples), 60)

    def test_backfilled_visits_after_cutoff_do_not_train(self):
        tables = fixture_tables()
        cutoff = dt.datetime(2025, 7, 16, tzinfo=dt.timezone.utc)
        original, _ = label_examples(tables, Conditions(tables, cutoff.date()), cutoff)
        late = copy.deepcopy(tables)
        changed_known_at = 0
        for visit in late["visits"]:
            if dt.date.fromisoformat(visit["visited_on"]) > cutoff.date():
                visit["known_at"] = "2025-07-18T00:00:00+00:00"
                visit["status"] = "not_workable"
                visit["work_start"] = visit["work_end"] = None
                changed_known_at += 1
        self.assertGreater(changed_known_at, 0)
        changed, audit = label_examples(late, Conditions(late, cutoff.date()), cutoff)
        self.assertEqual(len(original), len(changed))
        self.assertGreater(audit.get("unknown_at_cutoff", 0), 0)

    def test_interval_crossing_cutoff_is_excluded(self):
        tables = fixture_tables()
        cutoff = dt.datetime(2025, 7, 16, tzinfo=dt.timezone.utc)
        for visit in tables["visits"]:
            if visit["status"] == "worked":
                visit["work_start"] = "2025-07-15"
                visit["work_end"] = "2025-07-17"
                visit["known_at"] = "2025-07-10T00:00:00+00:00"
        examples, audit = label_examples(tables, Conditions(tables, cutoff.date()), cutoff)
        self.assertEqual(0, sum(1 for example in examples if example["label"] == 1))
        self.assertGreater(audit["overlaps_cutoff"], 0)


class ValidationLeakageTests(unittest.TestCase):
    def test_forward_validation_purges_guard_band_and_same_event(self):
        tables = fixture_tables()
        origin = dt.date(2025, 7, 3)
        window = [origin + dt.timedelta(days=offset) for offset in range(1, 8)]
        injected = copy.deepcopy(tables["visits"][0])
        injected.update({"visit_id": "leak", "plot_id": "p0",
                         "visited_on": "2025-07-01", "known_at": "2025-07-02T00:00:00+00:00",
                         "status": "worked", "work_start": "2025-07-01",
                         "work_end": "2025-07-01"})
        tables["visits"].append(injected)
        examples, audit = label_examples(
            tables, Conditions(tables, origin), origin,
            purge=_purge(Conditions(tables, origin), ["p0"], window))
        self.assertNotIn("leak", {example["visit_id"] for example in examples})
        self.assertGreater(audit.get("validation_purge", 0), 0)

    def test_forward_validation_reports_purge_rule(self):
        result = forward_validation(fixture_tables())
        self.assertIn("前向时间切分", result["scheme"])
        self.assertIn("同场降雨", result["scheme"])

    def test_future_conditions_do_not_change_training_examples(self):
        tables = fixture_tables()
        cutoff = dt.datetime(2025, 7, 16, tzinfo=dt.timezone.utc)
        original, _ = label_examples(tables, Conditions(tables, cutoff.date()), cutoff)
        revised = copy.deepcopy(tables)
        for row in revised["conditions"]:
            if dt.date.fromisoformat(row["day"]) > cutoff.date():
                row["rain_mm"] = 500.0
                row["soil_wetness"] = 0.99
        changed, _ = label_examples(revised, Conditions(revised, cutoff.date()), cutoff)
        self.assertEqual([(e["visit_id"], e["label"]) for e in original],
                         [(e["visit_id"], e["label"]) for e in changed])
        self.assertEqual([e["features"] for e in original], [e["features"] for e in changed])

    def test_validation_reports_budgets_not_single_accuracy(self):
        result = forward_validation(fixture_tables())
        self.assertIn("budgets", result)
        self.assertNotIn("accuracy", result)
        for row in result["budgets"].values():
            self.assertIn("mean_credible_windows", row)
            self.assertIn("plausible_share_among_verified", row)


class AdviceTests(unittest.TestCase):
    def test_fixture_without_signal_does_not_issue_precise_schedule(self):
        result = plan(fixture_tables())
        advice = result["advice"]
        self.assertTrue(advice)
        self.assertTrue(all(entry["advice"] in {"baseline", "withheld"} for entry in advice))
        self.assertTrue(all(entry["windows"] == [] for entry in advice))

    def test_learnable_signal_produces_ranked_windows(self):
        tables = rebuild()
        result = plan(tables, validation={})
        ranked = [entry for entry in result["advice"] if entry["advice"] == "ranked"]
        self.assertTrue(ranked)
        entry = ranked[0]
        self.assertLessEqual(len(entry["windows"]), entry["visit_budget"])
        for window in entry["windows"]:
            self.assertGreaterEqual(window["probability"], window["lower"])
            self.assertGreaterEqual(window["upper"], window["probability"])
        ladder = result["budget_ladder"]
        self.assertEqual(sorted(row["credible_windows"] for row in ladder),
                         [row["credible_windows"] for row in ladder])

    def test_extreme_rainfall_withholds_schedule(self):
        tables = rebuild()
        target = tables["requests"][0]["plot_id"]
        for row in tables["conditions"]:
            if row["plot_id"] == target and "2025-07-17" <= row["day"] <= "2025-07-23":
                row["rain_mm"] = 80.0
                row["soil_wetness"] = 0.99
        conditions = Conditions(tables, dt.date(2025, 7, 16))
        examples, _ = label_examples(tables, conditions, dt.datetime(2025, 7, 16,
                                                                   tzinfo=dt.timezone.utc))
        model = fit_model(examples, conditions.feature_names)
        advice = build_advice(model, conditions,
                              [tables["requests"][0]], dt.date(2025, 7, 16))["advice"][0]
        self.assertEqual("withheld", advice["advice"])
        self.assertTrue(any("超过" in reason for reason in advice["reasons"]))
        self.assertEqual([], advice["windows"])

    def test_single_crop_region_withholds(self):
        tables = rebuild()
        for row in tables["conditions"]:
            row["crop"] = "rice"
        result = plan(tables, validation={})
        self.assertTrue(all(entry["advice"] == "withheld" for entry in result["advice"]))
        self.assertTrue(any("可用作物只有 1 种" in reason
                            for entry in result["advice"] for reason in entry["reasons"]))

    def test_budget_ladder_counts_credible_windows_monotonically(self):
        tables = rebuild()
        result = plan(tables, validation={})
        ladder = budget_ladder({"advice": result["advice"]})
        credible = [row["credible_windows"] for row in ladder]
        self.assertEqual(credible, sorted(credible))
        self.assertTrue(all(row["withheld_requests"] == sum(
            1 for entry in result["advice"] if entry["advice"] == "withheld") for row in ladder))

    def test_reports_are_reproducible(self):
        tables = rebuild()
        first = plan(tables, validation={})
        second = plan(tables, validation={})
        self.assertEqual(json.dumps(first, sort_keys=True, default=str),
                         json.dumps(second, sort_keys=True, default=str))


if __name__ == "__main__":
    unittest.main()
