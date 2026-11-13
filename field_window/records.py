"""按地点与日期记录作业条件。"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class FieldDay:
    village_id: str
    day: date
    rain_mm: float | None
    workable: bool | None


def complete(record: FieldDay) -> bool:
    return record.rain_mm is not None and record.workable is not None
