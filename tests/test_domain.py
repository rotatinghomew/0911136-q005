"""验证现有领域摘要的业务边界。"""
import copy
import json
import unittest
from pathlib import Path

from field_window.analysis import visit_profile


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.tables = json.loads((Path(__file__).resolve().parents[1]/'fixtures/dataset.json').read_text())['tables']

    def test_work_interval_contains_both_ends(self):
        tables = copy.deepcopy(self.tables)
        tables['visits'] = [dict(tables['visits'][0],status='worked',work_start='2025-06-01',work_end='2025-06-03')]
        self.assertEqual(3,visit_profile(tables)['worked_intervals'][0]['possible_days'])

    def test_report_date_is_not_assumed_work_date(self):
        tables = copy.deepcopy(self.tables)
        first = visit_profile(tables)
        for row in tables['visits']:
            row['known_at'] = '2026-01-01T00:00:00+00:00'
        self.assertEqual(first,visit_profile(tables))

    def test_worked_without_bounds_fails(self):
        tables = copy.deepcopy(self.tables)
        tables['visits'][0].update(status='worked',work_start=None)
        with self.assertRaises(ValueError):
            visit_profile(tables)

    def test_unsupported_status_fails(self):
        tables = copy.deepcopy(self.tables)
        tables['visits'][0]['status'] = 'maybe'
        with self.assertRaises(ValueError):
            visit_profile(tables)
