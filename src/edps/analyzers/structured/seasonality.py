import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncGenerator, Optional, Tuple
from warnings import warn

import numpy as np
from extended_dataset_profile import TimeBasedGraph, Trend
from matplotlib.figure import Figure
from pandas import DataFrame, Series
from scipy.stats import linregress
from statsmodels.tsa.seasonal import STL, DecomposeResult

from edps.analyzers.structured.result_keys import (
    NUMERIC_GRAPH_ORIGINAL,
    NUMERIC_GRAPH_RESIDUAL,
    NUMERIC_GRAPH_SEASONALITY,
    NUMERIC_GRAPH_TREND,
    NUMERIC_TREND,
)
from edps.analyzers.structured.temporal_consistency import DatetimeColumnTemporalConsistency, Granularities, Granularity
from edps.analyzers.structured.type_parser import ColumnsWrapper, DatetimeColumnInfo, DatetimeKind
from edps.filewriter import get_pyplot_writer
from edps.taskcontext import TaskContext
from edps.types import SeasonalityConfig


class PerTimeBaseSeasonalityGraphs:
    def __init__(
        self,
        original: TimeBasedGraph,
        trend: TimeBasedGraph,
        seasonality: TimeBasedGraph,
        residual: TimeBasedGraph,
    ) -> None:
        self.original = original
        self.trend = trend
        self.seasonality = seasonality
        self.residual = residual


async def seasonal_decompose(
    ctx: TaskContext,
    datetime_column_infos: ColumnsWrapper[DatetimeColumnInfo],
    datetime_column_fields: DataFrame,
    numeric_columns: DataFrame,
    numeric_columns_iqr: Series,
) -> DataFrame:
    decompose_results = await _seasonal_decompose_numeric_over_datetime(
        ctx, datetime_column_infos, datetime_column_fields, numeric_columns
    )
    column_fields = await _get_seasonality_graphs(ctx, decompose_results)
    column_fields[NUMERIC_TREND] = await _compute_trends(ctx, decompose_results, numeric_columns_iqr)
    return column_fields


async def _seasonal_decompose_numeric_over_datetime(
    ctx: TaskContext,
    datetime_column_infos: ColumnsWrapper[DatetimeColumnInfo],
    datetime_column_fields: DataFrame,
    numeric_columns: DataFrame,
) -> DataFrame:
    datetime_count = len(datetime_column_infos.ids)
    row_count = len(numeric_columns.index)
    ctx.logger.info("Starting seasonality analysis on %d rows over %d time bases", row_count, datetime_count)

    dataframe = DataFrame(index=numeric_columns.columns, columns=datetime_column_infos.ids, dtype=object)

    for datetime_column_name, datetime_fields in datetime_column_fields.T.items():
        temporal_consistency: DatetimeColumnTemporalConsistency = datetime_fields["temporal-consistency"]
        datetime_column_duration: timedelta = datetime_fields["latest"] - datetime_fields["earliest"]
        datetime_kind = datetime_column_infos.get_info(str(datetime_column_name)).kind

        if datetime_kind == DatetimeKind.DATE:
            message = (
                f'Column "{datetime_column_name}" has date only format. This might impede the seasonality analysis. '
                "To fix this, supply it as datetime, preferably in the ISO8601 format."
            )
            warn(message)
            ctx.logger.warning(message)
        elif datetime_kind == DatetimeKind.TIME:
            message = (
                f'Column "{datetime_column_name}" has time only format. It will be skipped for seasonality analysis'
            )
            warn(message)
            ctx.logger.warning(message)
            continue

        filtered_numeric_columns = numeric_columns.loc[temporal_consistency.cleaned_series.index].set_index(
            temporal_consistency.cleaned_series, inplace=False
        )

        for numeric_column_name, numeric_column in filtered_numeric_columns.items():
            resample_period = _get_biggest_fitting_period(
                datetime_column_duration,
                numeric_column.count(),
                temporal_consistency,
                ctx.config.structured_config.seasonality,
            )

            dataframe.loc[str(numeric_column_name), str(datetime_column_name)] = _seasonal_decompose_column(
                ctx, numeric_column, resample_period, str(datetime_column_name)
            )

    ctx.logger.info("Finished seasonality analysis.")
    return dataframe


async def _get_seasonality_graphs(ctx: TaskContext, decompose_results: DataFrame) -> DataFrame:
    dataframe = DataFrame(
        None,
        index=decompose_results.index,
        columns=[NUMERIC_GRAPH_ORIGINAL, NUMERIC_GRAPH_SEASONALITY, NUMERIC_GRAPH_TREND, NUMERIC_GRAPH_RESIDUAL],
    )
    for numeric_column_name, decompose_results_for_numeric_column in decompose_results.T.items():
        numeric_column_name = str(numeric_column_name)
        graphs = [
            await _get_seasonality_graphs_single_cell(
                ctx,
                numeric_column_name,
                str(datetime_column_name),
                decompose_result_for_cell,
            )
            for datetime_column_name, decompose_result_for_cell in decompose_results_for_numeric_column.items()
            if decompose_result_for_cell is not None
        ]
        dataframe.loc[numeric_column_name, NUMERIC_GRAPH_ORIGINAL] = [
            single_cell_graphs.original for single_cell_graphs in graphs
        ]
        dataframe.loc[numeric_column_name, NUMERIC_GRAPH_SEASONALITY] = [
            single_cell_graphs.seasonality for single_cell_graphs in graphs
        ]
        dataframe.loc[numeric_column_name, NUMERIC_GRAPH_TREND] = [
            single_cell_graphs.trend for single_cell_graphs in graphs
        ]
        dataframe.loc[numeric_column_name, NUMERIC_GRAPH_RESIDUAL] = [
            single_cell_graphs.residual for single_cell_graphs in graphs
        ]

    return dataframe


async def _get_seasonality_graphs_single_cell(
    ctx: TaskContext,
    column_name: str,
    time_base_column_name: str,
    seasonality: DecomposeResult,
) -> PerTimeBaseSeasonalityGraphs:
    xlim = seasonality._observed.index[0], seasonality._observed.index[-1]

    @asynccontextmanager
    async def get_plot(plot_type: str):
        plot_name = ctx.build_output_reference(f"{column_name}_over_{time_base_column_name}_{plot_type.lower()}")
        async with get_pyplot_writer(ctx, plot_name) as (axes, reference):
            axes.set_title(f"{plot_type} of {column_name} over {time_base_column_name}")
            axes.set_xlabel(time_base_column_name)
            axes.set_ylabel(f"{plot_type} of {column_name}")
            if axes.figure:
                axes.set_xlim(xlim)
            if isinstance(axes.figure, Figure):
                axes.figure.set_figwidth(20)
            yield axes, reference

    async with get_plot("Original") as (axes, original_reference):
        axes.plot(seasonality.observed)

    async with get_plot("Trend") as (axes, trend_reference):
        axes.plot(seasonality.trend)

    async with get_plot("Seasonality") as (axes, seasonal_reference):
        axes.plot(seasonality.seasonal)

    async with get_plot("Residual") as (axes, residual_reference):
        axes.plot(seasonality.resid, marker="o", linestyle="none")
        axes.plot(xlim, (0, 0), zorder=-3)

    return PerTimeBaseSeasonalityGraphs(
        original=TimeBasedGraph(timeBaseColumn=time_base_column_name, file=original_reference),
        trend=TimeBasedGraph(timeBaseColumn=time_base_column_name, file=trend_reference),
        seasonality=TimeBasedGraph(timeBaseColumn=time_base_column_name, file=seasonal_reference),
        residual=TimeBasedGraph(timeBaseColumn=time_base_column_name, file=residual_reference),
    )


def _seasonal_decompose_column(
    ctx: TaskContext, column: Series, resample_period: Granularity, datetime_column_name: str
) -> Optional[DecomposeResult]:
    non_null_column: Series = column[column.notnull()]
    number_non_null = len(non_null_column)
    non_null_column = (
        non_null_column.resample(
            resample_period.timedelta,
        )
        .mean()
        .interpolate()
    )
    resample_period_samples = resample_period.samples_per_period
    number_non_null = len(non_null_column.index)

    number_periods = number_non_null / resample_period_samples
    # Check if we have enough samples to run seasonal decompose.
    if number_periods < 2.0:
        ctx.logger.info(
            'Column "%s" contains only %d samples when resampled to %s for running seasonality analysis. Need at least %d samples.',
            column.name,
            number_non_null,
            resample_period,
            2 * resample_period_samples,
        )
        return None

    try:
        stl = STL(non_null_column, period=resample_period_samples, robust=True)
        return stl.fit()
    except ValueError as error:
        message = f"Seasonal decompose of {column.name} over {datetime_column_name} failed: {error}"
        ctx.logger.warning(message)
        warn(message)
        return None


_LARGEST_TO_LOWEST_GRANULARITY = sorted(
    (granularity.value for granularity in Granularities),
    reverse=True,
    key=lambda granularity: granularity.timedelta,
)


def _get_biggest_fitting_period(
    duration: timedelta,
    number_non_null: int,
    temporal_consistency: DatetimeColumnTemporalConsistency,
    config: SeasonalityConfig,
) -> Granularity:
    target_count = min(number_non_null, config.target_samples)
    for granularity in _LARGEST_TO_LOWEST_GRANULARITY:
        number_of_samples_after_resample = duration / granularity.timedelta
        if number_of_samples_after_resample >= target_count:
            return granularity

    return temporal_consistency.main_period


async def _compute_trends(ctx: TaskContext, decompose_results: DataFrame, numeric_columns_iqr: Series) -> Series:
    return Series(
        {
            numeric_column_name: trend
            async for numeric_column_name, trend in _iterate_trend(
                ctx.config.structured_config.seasonality, decompose_results, numeric_columns_iqr
            )
        },
        name=NUMERIC_TREND,
    )


async def _iterate_trend(
    config: SeasonalityConfig, decompose_results: DataFrame, numeric_columns_iqr: Series
) -> AsyncGenerator[Tuple[str, Trend], None]:
    for numeric_column_name, decompose_results_for_column in decompose_results.T.items():
        numeric_column_name = str(numeric_column_name)
        trends = [
            await calculate_trend(config, decompose_result.trend, numeric_columns_iqr[numeric_column_name])
            for decompose_result in decompose_results_for_column
        ]
        if len(trends) == 0 or all(trend == Trend.NoTrend for trend in trends):
            yield numeric_column_name, Trend.NoTrend
        elif all(trend == Trend.Increasing or trend == Trend.NoTrend for trend in trends):
            yield numeric_column_name, Trend.Increasing
        elif all(trend == Trend.Decreasing or trend == Trend.NoTrend for trend in trends):
            yield numeric_column_name, Trend.Decreasing
        else:
            # Detected multiple different trends, depending on time base.
            yield numeric_column_name, Trend.NoTrend


async def calculate_trend(config: SeasonalityConfig, series: Series, iqr: float) -> Trend:
    result = await asyncio.to_thread(linregress, np.arange(0, len(series)), series.to_numpy())
    if iqr == 0.0:
        return Trend.NoTrend
    normalized_slope = result.slope / iqr
    if normalized_slope >= config.trend_threshold:
        return Trend.Increasing
    elif normalized_slope <= -config.trend_threshold:
        return Trend.Decreasing
    else:
        return Trend.NoTrend
