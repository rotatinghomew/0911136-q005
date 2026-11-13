import unittest
from datetime import date

from field_window.records import FieldDay, complete


class FieldTests(unittest.TestCase):
    def test_missing_weather_is_not_complete(self):
        self.assertFalse(complete(FieldDay("v1", date(2026, 8, 2), None, True)))
