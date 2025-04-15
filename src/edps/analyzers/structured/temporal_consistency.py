from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from warnings import warn

from extended_dataset_profile import TemporalConsistency
from numpy import count_nonzero
from pandas import DataFrame, Series, Timedelta, UInt64Dtype

from edps.taskcontext import TaskContext


@dataclass(frozen=True)
class Granularity:
    name: str
    timedelta: Timedelta
    samples_per_period: int

    def __str__(self) -> str:
        return self.name


class Granularities(Enum):
    Microseconds = Granularity(name="microseconds", timedelta=Timedelta(microseconds=1), samples_per_period=1000)
    Milliseconds = Granularity(name="milliseconds", timedelta=Timedelta(milliseconds=1), samples_per_period=1000)
    Seconds = Granularity(name="seconds", timedelta=Timedelta(seconds=1), samples_per_period=60)
    Minutes = Granularity(name="minutes", timedelta=Timedelta(minutes=1), samples_per_period=60)
    Hours = Granularity(name="hours", timedelta=Timedelta(hours=1), samples_per_period=24)
    Days = Granularity(name="days", timedelta=Timedelta(days=1), samples_per_period=7)
    Weeks = Granularity(name="weeks", timedelta=Timedelta(weeks=1), samples_per_period=4)
    Months = Granularity(name="months", timedelta=Timedelta(days=30), samples_per_period=12)
    Years = Granularity(name="years", timedelta=Timedelta(days=365), samples_per_period=10)


class DatetimeColumnTemporalConsistency:
    def __init__(
        self,
        main_period: Granularity,
        temporal_consistencies: List[TemporalConsistency],
        cleaned_series: Series,
    ) -> None:
        self.main_period = main_period
        self.temporal_consistencies = temporal_consistencies
        self.cleaned_series = cleaned_series

    def get_main_temporal_consistency(self) -> TemporalConsistency:
        return self.__getitem__(self.main_period.name)

    def __getitem__(self, period: str) -> TemporalConsistency:
        try:
            return next((consistency for consistency in self.temporal_consistencies if consistency.timeScale == period))
        except StopIteration as error:
            raise KeyError(f"{period} not in the temporal consistencies") from error


async def compute_temporal_consistency(ctx: TaskContext, columns: DataFrame) -> Series:
    return Series({name: _compute_temporal_consistency_for_column(ctx, column) for name, column in columns.items()})


def determine_periodicity(gaps: Series, distincts: Series):
    diff = distincts - gaps
    return diff.idxmax()


def _compute_temporal_consistency_for_column(
    ctx: TaskContext, column: Series
) -> Optional[DatetimeColumnTemporalConsistency]:
    column = column.sort_values(ascending=True)
    row_count = len(column)

    TIME_BASE_THRESHOLD = 15
    unique_timestamps = column.nunique()
    if unique_timestamps < TIME_BASE_THRESHOLD:
        message = (
            "Can not analyze temporal consistency, time base contains too few unique timestamps. "
            f"Have {unique_timestamps}, need at least {TIME_BASE_THRESHOLD}."
        )
        ctx.logger.warning(message)
        warn(message)
        return None

    # Remove null entries
    column = column[column.notnull()]  # type: ignore
    new_count = len(column)
    if new_count < row_count:
        empty_index_count = row_count - new_count
        message = f"Filtered out {empty_index_count} rows, because their index was empty"
        ctx.logger.warning(message)
        warn(message)
        row_count = new_count

    deltas = column.diff()[1:]
    gaps = Series(
        {granularity.value: count_nonzero(deltas > granularity.value.timedelta) for granularity in Granularities},
        dtype=UInt64Dtype(),
    )
    distincts = Series(
        {granularity.value: column.dt.round(granularity.value.timedelta).nunique() for granularity in Granularities},  # type: ignore
        dtype=UInt64Dtype(),
    )
    stable = distincts == 1
    periodicity: Granularity = determine_periodicity(gaps, distincts)
    temporal_consistencies = [
        TemporalConsistency(
            timeScale=granularity.value.name,
            differentAbundancies=distincts[granularity.value],  # type: ignore
            stable=stable[granularity.value],  # type: ignore
            numberOfGaps=gaps[granularity.value],  # type: ignore
        )
        for granularity in Granularities
    ]

    return DatetimeColumnTemporalConsistency(periodicity, temporal_consistencies, column)
