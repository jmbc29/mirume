"""SM-2 spaced-repetition scheduling for Mirume.

Implements the classic SuperMemo SM-2 algorithm used to schedule review of
:class:`models.SavedWord` (and :class:`models.SavedGrammar`) rows. Given the
current scheduling state of an item and a 0-5 self-graded recall quality,
:func:`sm2` returns the next state: a new ease factor, interval and
repetition count, plus the concrete next review date.

Grading scale (standard SM-2):

* 0-2 — failed recall. Repetitions reset to 0 and the item is shown again
  the next day.
* 3-5 — successful recall. The interval grows (1 day -> 6 days -> previous
  interval * ease factor), and the ease factor is nudged up or down based on
  how easy the recall felt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

#: SM-2 ease factor never drops below this floor, or reviews would spiral
#: into ever-shorter intervals for a merely "hard" (not failed) item.
MIN_EASE_FACTOR: float = 1.3

#: Ease factor assigned to a brand-new item.
DEFAULT_EASE_FACTOR: float = 2.5

#: Grades below this are treated as a failed recall.
PASS_THRESHOLD: int = 3


@dataclass(frozen=True, slots=True)
class SM2Result:
    """The updated scheduling state for one item after a review.

    Attributes:
        ease_factor: New ease factor (>= :data:`MIN_EASE_FACTOR`).
        interval_days: New interval, in days, until the next review.
        repetitions: New count of consecutive successful reviews.
        next_review_date: Timezone-aware UTC datetime the item is next due.
    """

    ease_factor: float
    interval_days: int
    repetitions: int
    next_review_date: datetime


def sm2(
    ease_factor: float,
    interval_days: int,
    repetitions: int,
    grade: int,
    *,
    now: datetime | None = None,
) -> SM2Result:
    """Compute the next SM-2 scheduling state after a review.

    Args:
        ease_factor: Current ease factor (2.5 for a never-reviewed item).
        interval_days: Current interval in days (0 for a never-reviewed item).
        repetitions: Current count of consecutive successful reviews.
        grade: Self-graded recall quality, 0 (total blackout) to 5 (perfect
            recall). Values are clamped into ``[0, 5]``.
        now: Reference time for computing ``next_review_date``. Defaults to
            the current UTC time; overridable for testing.

    Returns:
        The updated :class:`SM2Result`.
    """
    grade = max(0, min(5, grade))
    now = now or datetime.now(timezone.utc)

    if grade < PASS_THRESHOLD:
        new_repetitions = 0
        new_interval = 1
    else:
        new_repetitions = repetitions + 1
        if new_repetitions == 1:
            new_interval = 1
        elif new_repetitions == 2:
            new_interval = 6
        else:
            new_interval = round(interval_days * ease_factor)

    new_ease_factor = ease_factor + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))
    new_ease_factor = max(MIN_EASE_FACTOR, new_ease_factor)

    return SM2Result(
        ease_factor=round(new_ease_factor, 3),
        interval_days=new_interval,
        repetitions=new_repetitions,
        next_review_date=now + timedelta(days=new_interval),
    )


if __name__ == "__main__":
    # Simulate 10 review sessions for one item and print the interval
    # progression, alternating a couple of grades to show both branches.
    ease_factor, interval_days, repetitions = DEFAULT_EASE_FACTOR, 0, 0
    grades = [5, 4, 5, 5, 2, 5, 4, 5, 3, 5]

    print(f"{'session':>7}  {'grade':>5}  {'ease':>5}  {'interval':>8}  {'reps':>4}")
    for session, grade in enumerate(grades, start=1):
        result = sm2(ease_factor, interval_days, repetitions, grade)
        ease_factor, interval_days, repetitions = (
            result.ease_factor,
            result.interval_days,
            result.repetitions,
        )
        print(
            f"{session:>7}  {grade:>5}  {ease_factor:>5.2f}  "
            f"{interval_days:>7}d  {repetitions:>4}"
        )
