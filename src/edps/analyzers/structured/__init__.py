import itertools
from collections.abc import Hashable
from datetime import timedelta
from enum import Enum
from pathlib import PurePosixPath
from typing import Dict, Optional, Tuple, cast
from warnings import warn

import pandas as pd
from extended_dataset_profile import (
    CorrelationSummary,
    DateTimeColumn,
    FileReference,
    NumericColumn,
    StringColumn,
    StructuredDataSet,
    TemporalCover,
)
from matplotlib.figure import Figure
from matplotlib.pyplot import get_cmap
from numpy import linspace, ones_like, triu
from pandas import (
    DataFrame,
    Index,
    RangeIndex,
    Series,
    concat,
)
from scipy.stats import distributions
from seaborn import heatmap

from edps.analyzers.structured.result_keys import (
    COMMON_INCONSISTENT,
    COMMON_INTERPRETABLE,
    COMMON_NULL,
    COMMON_UNIQUE,
    DATETIME_ALL_ENTRIES_UNIQUE,
    DATETIME_EARLIEST,
    DATETIME_LATEST,
    DATETIME_MONOTONIC_DECREASING,
    DATETIME_MONOTONIC_INCREASING,
    DATETIME_TEMPORAL_CONSISTENCY,
    NUMERIC_DISTRIBUTION,
    NUMERIC_DISTRIBUTION_PARAMETERS,
    NUMERIC_GRAPH_ORIGINAL,
    NUMERIC_GRAPH_RESIDUAL,
    NUMERIC_GRAPH_SEASONALITY,
    NUMERIC_GRAPH_TREND,
    NUMERIC_IQR,
    NUMERIC_IQR_OUTLIERS,
    NUMERIC_LOWER_DIST,
    NUMERIC_LOWER_IQR,
    NUMERIC_LOWER_PERCENTILE,
    NUMERIC_LOWER_QUANT,
    NUMERIC_LOWER_Z,
    NUMERIC_MAX,
    NUMERIC_MEAN,
    NUMERIC_MEDIAN,
    NUMERIC_MIN,
    NUMERIC_PERCENTILE_OUTLIERS,
    NUMERIC_QUANT_OUTLIERS,
    NUMERIC_RELATIVE_OUTLIERS,
    NUMERIC_STD_DEV,
    NUMERIC_TREND,
    NUMERIC_UPPER_DIST,
    NUMERIC_UPPER_IQR,
    NUMERIC_UPPER_PERCENTILE,
    NUMERIC_UPPER_QUANT,
    NUMERIC_UPPER_Z,
    NUMERIC_VARIANCE,
    NUMERIC_Z_OUTLIERS,
)
from edps.analyzers.structured.seasonality import seasonal_decompose
from edps.filewriter import get_pyplot_writer
from edps.taskcontext import TaskContext

from .fitter import DistributionParameters, FittingError, Limits, fit_best_distribution
from .temporal_consistency import DatetimeColumnTemporalConsistency, compute_temporal_consistency
from .temporal_consistency import determine_periodicity as determine_periodicity
from .type_parser import (
    ColumnsWrapper,
    DatetimeColumnInfo,
    Result,
    parse_types,
)


class _Distributions(str, Enum):
    SingleValue = "single value"
    TooSmallDataset = "dataset too small to determine distribution"
    NoMatch = "could not match any distribution"


class PandasAnalyzer:
    def __init__(self, data: DataFrame):
        self._data = data
        self._max_elements_per_column = 100000
        self._intervals = [
            timedelta(seconds=1),
            timedelta(minutes=1),
            timedelta(hours=1),
            timedelta(days=1),
            timedelta(weeks=1),
        ]

    async def analyze(self, ctx: TaskContext) -> StructuredDataSet:
        row_count = len(self._data.index)
        ctx.logger.info(
            "Started structured data analysis with dataset containing %d rows",
            row_count,
        )

        type_parser_results = parse_types(ctx, self._data)

        all_cols = type_parser_results.all_cols
        common_fields = await self._compute_common_fields(all_cols.data, type_parser_results)

        datetime_cols = type_parser_results.datetime_cols
        datetime_common_fields = common_fields.loc[Index(datetime_cols.ids)]
        datetime_fields = await self._compute_datetime_fields(ctx, datetime_cols.data)
        datetime_fields = concat([datetime_common_fields, datetime_fields], axis=1)

        numeric_cols = type_parser_results.numeric_cols
        numeric_common_fields = common_fields.loc[Index(numeric_cols.ids)]
        numeric_fields = await self._compute_numeric_fields(
            ctx, numeric_cols.data, numeric_common_fields, datetime_cols, datetime_fields
        )
        numeric_fields = concat([numeric_fields, numeric_common_fields], axis=1)

        string_cols = type_parser_results.string_cols
        string_fields = common_fields.loc[Index(string_cols.ids)]

        transformed_numeric_columns = [
            await self._transform_numeric_results(
                ctx,
                numeric_cols.get_col(id),
                _get_single_row(id, numeric_fields),
            )
            for id in numeric_cols.ids
        ]
        transformed_datetime_columns = [
            await self._transform_datetime_results(
                ctx,
                datetime_cols.get_col(id),
                _get_single_row(id, datetime_fields),
                datetime_cols.get_info(id),
            )
            for id in datetime_cols.ids
        ]
        transformed_string_columns = [
            await self._transform_string_results(ctx, string_cols.get_col(id), _get_single_row(id, string_fields))
            for id in string_cols.ids
        ]

        correlation_ids = numeric_cols.ids
        correlation_columns = all_cols.data.loc[:, correlation_ids]
        correlation_fields = common_fields.loc[Index(correlation_ids)]
        correlation_matrix = await _get_correlation_matrix(
            ctx,
            correlation_columns,
            correlation_fields,
        )
        correlation_graph = await _get_correlation_graph(
            ctx, ctx.build_output_reference("correlations"), correlation_matrix
        )
        correlation_summary = await _get_correlation_summary(ctx, correlation_matrix)

        transformed_numeric_column_count = len(transformed_numeric_columns)
        transformed_date_time_column_count = len(transformed_datetime_columns)
        transformed_string_column_count = len(transformed_string_columns)
        column_count = (
            transformed_numeric_column_count + transformed_date_time_column_count + transformed_string_column_count
        )

        return StructuredDataSet(
            rowCount=row_count,
            columnCount=column_count,
            numericColumnCount=transformed_numeric_column_count,
            datetimeColumnCount=transformed_date_time_column_count,
            stringColumnCount=transformed_string_column_count,
            numericColumns=transformed_numeric_columns,
            datetimeColumns=transformed_datetime_columns,
            stringColumns=transformed_string_columns,
            correlationGraph=correlation_graph,
            correlationSummary=correlation_summary,
        )

    async def _compute_common_fields(self, columns: DataFrame, type_parser_results: Result) -> DataFrame:
        common_fields = DataFrame(index=columns.columns)
        common_fields[COMMON_NULL] = Series(
            {name: info.number_nan_before_conversion for name, info, _ in type_parser_results.all_cols},
            name=COMMON_NULL,
        )
        common_fields[COMMON_INCONSISTENT] = Series(
            {name: info.number_inconsistent for name, info, _ in type_parser_results.all_cols},
            name=COMMON_INCONSISTENT,
        )
        common_fields[COMMON_INTERPRETABLE] = Series(
            {name: info.number_interpretable for name, info, _ in type_parser_results.all_cols},
            name=COMMON_INTERPRETABLE,
        )
        common_fields[COMMON_UNIQUE] = columns.nunique(dropna=True)
        return common_fields

    async def _compute_numeric_fields(
        self,
        ctx: TaskContext,
        columns: DataFrame,
        common_fields: DataFrame,
        datetime_column_infos: ColumnsWrapper[DatetimeColumnInfo],
        datetime_fields: DataFrame,
    ) -> DataFrame:
        fields = DataFrame(index=columns.columns)

        fields[NUMERIC_MIN] = columns.min()
        fields[NUMERIC_MAX] = columns.max()
        fields[NUMERIC_LOWER_PERCENTILE] = columns.quantile(0.01)
        fields[NUMERIC_UPPER_PERCENTILE] = columns.quantile(0.99)
        fields[NUMERIC_PERCENTILE_OUTLIERS] = _get_outliers(
            columns,
            fields[NUMERIC_LOWER_PERCENTILE],
            fields[NUMERIC_UPPER_PERCENTILE],
        )
        # Standard Distribution
        fields[NUMERIC_MEAN] = columns.mean()
        fields[NUMERIC_MEDIAN] = columns.median()
        fields[NUMERIC_VARIANCE] = columns.var()
        fields[NUMERIC_STD_DEV] = columns.std()
        fields[NUMERIC_LOWER_Z] = fields[NUMERIC_MEAN] - 3.0 * fields[NUMERIC_STD_DEV]
        fields[NUMERIC_UPPER_Z] = fields[NUMERIC_MEAN] + 3.0 * fields[NUMERIC_STD_DEV]
        fields[NUMERIC_Z_OUTLIERS] = _get_outliers(columns, fields[NUMERIC_LOWER_Z], fields[NUMERIC_UPPER_Z])
        # Inter Quartile Range
        fields[NUMERIC_LOWER_QUANT] = columns.quantile(0.25)
        fields[NUMERIC_UPPER_QUANT] = columns.quantile(0.75)
        fields[NUMERIC_QUANT_OUTLIERS] = _get_outliers(
            columns, fields[NUMERIC_LOWER_QUANT], fields[NUMERIC_UPPER_QUANT]
        )
        fields[NUMERIC_IQR] = fields[NUMERIC_UPPER_QUANT] - fields[NUMERIC_LOWER_QUANT]
        fields[NUMERIC_LOWER_IQR] = fields[NUMERIC_LOWER_QUANT] - 1.5 * fields[NUMERIC_IQR]
        fields[NUMERIC_UPPER_IQR] = fields[NUMERIC_UPPER_QUANT] + 1.5 * fields[NUMERIC_IQR]
        fields[NUMERIC_IQR_OUTLIERS] = _get_outliers(columns, fields[NUMERIC_LOWER_IQR], fields[NUMERIC_UPPER_IQR])
        # Relative outliers
        fields[NUMERIC_RELATIVE_OUTLIERS] = (
            fields[[NUMERIC_PERCENTILE_OUTLIERS, NUMERIC_Z_OUTLIERS, NUMERIC_IQR_OUTLIERS]].mean(axis=1, skipna=True)
            / common_fields[COMMON_INTERPRETABLE]
        )
        # Distribution
        fields[NUMERIC_LOWER_DIST] = fields[NUMERIC_LOWER_IQR]
        fields.loc[fields[NUMERIC_LOWER_IQR] < fields[NUMERIC_MIN], NUMERIC_LOWER_DIST] = fields[NUMERIC_MIN]
        fields[NUMERIC_UPPER_DIST] = fields[NUMERIC_UPPER_IQR]
        fields.loc[fields[NUMERIC_UPPER_IQR] > fields[NUMERIC_MAX], NUMERIC_UPPER_DIST] = fields[NUMERIC_MAX]
        upper_equals_lower = fields[NUMERIC_LOWER_DIST] == fields[NUMERIC_UPPER_DIST]
        fields.loc[upper_equals_lower, NUMERIC_LOWER_DIST] = fields[NUMERIC_LOWER_DIST] * 0.9
        fields.loc[upper_equals_lower, NUMERIC_UPPER_DIST] = fields[NUMERIC_UPPER_DIST] * 1.1
        fields[NUMERIC_DISTRIBUTION], fields[NUMERIC_DISTRIBUTION_PARAMETERS] = await _get_distributions(
            ctx, columns, concat([common_fields, fields], axis=1)
        )
        seasonality_fields = await seasonal_decompose(
            ctx, datetime_column_infos, datetime_fields, columns, fields[NUMERIC_IQR]
        )
        fields = pd.concat([fields, seasonality_fields], axis=1)
        return fields

    async def _compute_datetime_fields(self, ctx: TaskContext, columns: DataFrame) -> DataFrame:
        computed = DataFrame(index=columns.columns)
        computed[DATETIME_EARLIEST] = columns.min()
        computed[DATETIME_LATEST] = columns.max()
        # TODO: Vectorize these
        computed[DATETIME_ALL_ENTRIES_UNIQUE] = Series({name: column.is_unique for name, column in columns.items()})
        computed[DATETIME_MONOTONIC_INCREASING] = Series(
            {name: column.is_monotonic_increasing for name, column in columns.items()}
        )
        computed[DATETIME_MONOTONIC_DECREASING] = Series(
            {name: column.is_monotonic_decreasing for name, column in columns.items()}
        )
        computed[DATETIME_TEMPORAL_CONSISTENCY] = await compute_temporal_consistency(ctx, columns)
        return computed

    async def _transform_numeric_results(
        self, ctx: TaskContext, column: Series, computed_fields: Series
    ) -> NumericColumn:
        ctx.logger.debug('Transforming numeric column "%s" results to EDP', column.name)
        box_plot = await _generate_box_plot(ctx, column)

        column_result = NumericColumn(
            name=str(column.name),
            nullCount=computed_fields[COMMON_NULL],
            inconsistentCount=computed_fields[COMMON_INCONSISTENT],
            interpretableCount=computed_fields[COMMON_INTERPRETABLE],
            numberUnique=computed_fields[COMMON_UNIQUE],
            min=computed_fields[NUMERIC_MIN],
            max=computed_fields[NUMERIC_MAX],
            mean=computed_fields[NUMERIC_MEAN],
            median=computed_fields[NUMERIC_MEDIAN],
            variance=computed_fields[NUMERIC_VARIANCE],
            stddev=computed_fields[NUMERIC_STD_DEV],
            upperPercentile=computed_fields[NUMERIC_UPPER_PERCENTILE],
            lowerPercentile=computed_fields[NUMERIC_LOWER_PERCENTILE],
            percentileOutlierCount=computed_fields[NUMERIC_PERCENTILE_OUTLIERS],
            upperQuantile=computed_fields[NUMERIC_UPPER_QUANT],
            lowerQuantile=computed_fields[NUMERIC_LOWER_QUANT],
            quantileOutlierCount=computed_fields[NUMERIC_QUANT_OUTLIERS],
            upperZScore=computed_fields[NUMERIC_UPPER_Z],
            lowerZScore=computed_fields[NUMERIC_LOWER_Z],
            zScoreOutlierCount=computed_fields[NUMERIC_Z_OUTLIERS],
            upperIQR=computed_fields[NUMERIC_UPPER_IQR],
            lowerIQR=computed_fields[NUMERIC_LOWER_IQR],
            iqr=computed_fields[NUMERIC_IQR],
            iqrOutlierCount=computed_fields[NUMERIC_IQR_OUTLIERS],
            relativeOutlierCount=computed_fields[NUMERIC_RELATIVE_OUTLIERS],
            distribution=computed_fields[NUMERIC_DISTRIBUTION],
            dataType=str(column.dtype),
            boxPlot=box_plot,
            original_series=computed_fields[NUMERIC_GRAPH_ORIGINAL],
            seasonalities=computed_fields[NUMERIC_GRAPH_SEASONALITY],
            trends=computed_fields[NUMERIC_GRAPH_TREND],
            residuals=computed_fields[NUMERIC_GRAPH_RESIDUAL],
            trend=computed_fields[NUMERIC_TREND],
        )
        distribution = computed_fields[NUMERIC_DISTRIBUTION]
        if (
            distribution not in [enum.value for enum in _Distributions]
            and column_result.numberUnique > ctx.config.structured_config.distribution.minimum_number_numeric_values
        ):
            column_result.distributionGraph = await _plot_distribution(
                ctx,
                column,
                computed_fields,
                computed_fields[NUMERIC_DISTRIBUTION],
                computed_fields[NUMERIC_DISTRIBUTION_PARAMETERS],
            )
        else:
            ctx.logger.debug(
                "Too few unique values for distribution analysis on column %s. Have %d unique, need at least %d.",
                column.name,
                column_result.numberUnique,
                ctx.config.structured_config.distribution.minimum_number_numeric_values,
            )

        return column_result

    async def _transform_datetime_results(
        self, ctx: TaskContext, column: Series, computed_fields: Series, info: DatetimeColumnInfo
    ) -> DateTimeColumn:
        ctx.logger.debug('Transforming datetime column "%s" results to EDP', column.name)
        temporal_consistency: Optional[DatetimeColumnTemporalConsistency] = computed_fields[
            DATETIME_TEMPORAL_CONSISTENCY
        ]

        return DateTimeColumn(
            name=str(column.name),
            nullCount=computed_fields[COMMON_NULL],
            inconsistentCount=computed_fields[COMMON_INCONSISTENT],
            interpretableCount=computed_fields[COMMON_INTERPRETABLE],
            numberUnique=computed_fields[COMMON_UNIQUE],
            temporalCover=TemporalCover(
                earliest=computed_fields[DATETIME_EARLIEST],
                latest=computed_fields[DATETIME_LATEST],
            ),
            all_entries_are_unique=computed_fields[DATETIME_ALL_ENTRIES_UNIQUE],
            monotonically_increasing=computed_fields[DATETIME_MONOTONIC_INCREASING],
            monotonically_decreasing=computed_fields[DATETIME_MONOTONIC_DECREASING],
            temporalConsistencies=(temporal_consistency.temporal_consistencies if temporal_consistency else []),
            periodicity=temporal_consistency.main_period.name if temporal_consistency else None,
            format=info.format,
        )

    async def _transform_string_results(
        self, ctx: TaskContext, column: Series, computed_fields: Series
    ) -> StringColumn:
        ctx.logger.debug('Transforming string column "%s" results to EDP', column.name)
        return StringColumn(
            name=str(column.name),
            nullCount=computed_fields[COMMON_NULL],
            inconsistentCount=computed_fields[COMMON_INCONSISTENT],
            interpretableCount=computed_fields[COMMON_INTERPRETABLE],
            numberUnique=computed_fields[COMMON_UNIQUE],
            distributionGraph=await _get_string_distribution_graph(ctx, column),
        )


async def _generate_box_plot(ctx: TaskContext, column: Series) -> FileReference:
    plot_name = ctx.build_output_reference(f"{column.name}_box_plot")
    async with get_pyplot_writer(ctx, plot_name) as (axes, reference):
        if isinstance(axes.figure, Figure):
            axes.figure.set_figwidth(3.0)
        axes.boxplot(column, notch=False, tick_labels=[str(column.name)])
    return reference


def _get_outliers(column: DataFrame, lower_limit: Series, upper_limit: Series) -> Series:
    is_outlier = (column < lower_limit) | (column > upper_limit)
    return is_outlier.sum()


async def _get_distributions(ctx: TaskContext, columns: DataFrame, fields: DataFrame) -> Tuple[Series, Series]:
    distributions: Dict[str, str] = {}
    parameters: Dict[str, DistributionParameters] = {}

    for index, column_name_and_data in enumerate(columns.items(), 1):
        column_name_hash, column_data = column_name_and_data
        column_name = str(column_name_hash)
        distribution, params = await _get_distribution(ctx, column_data, _get_single_row(column_name, fields))
        distributions[column_name] = distribution
        parameters[column_name] = params
        ctx.logger.debug("Computed %d/%d distributions", index, len(columns.columns))

    return Series(distributions, name=NUMERIC_DISTRIBUTION), Series(parameters, name=NUMERIC_DISTRIBUTION_PARAMETERS)


async def _get_distribution(ctx: TaskContext, column: Series, fields: Series) -> Tuple[str, DistributionParameters]:
    config = ctx.config.structured_config.distribution
    if fields[COMMON_UNIQUE] <= 1:
        return _Distributions.SingleValue.value, dict()

    if fields[COMMON_INTERPRETABLE] < config.minimum_number_numeric_values:
        return _Distributions.TooSmallDataset.value, dict()

    limits = Limits(min=fields[NUMERIC_LOWER_DIST], max=fields[NUMERIC_UPPER_DIST])
    try:
        return await fit_best_distribution(ctx, column, limits=limits)
    except FittingError as error:
        message = f'Distribution fitting for "{column.name}" failed: {error}'
        warn(message)
        ctx.logger.warning(message)
        return _Distributions.NoMatch, dict()


async def _plot_distribution(
    ctx: TaskContext,
    column: Series,
    column_fields: Series,
    distribution_name: str,
    distribution_parameters: dict,
):
    x_min = column_fields[NUMERIC_LOWER_DIST]
    x_max = column_fields[NUMERIC_UPPER_DIST]
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    x_limits = (x_min, x_max)
    plot_name = ctx.build_output_reference(f"{column.name}_distribution")
    async with get_pyplot_writer(ctx, plot_name) as (axes, reference):
        axes.set_title(f"Distribution of {column.name}")
        axes.set_xlabel(f"Value of {column.name}")
        axes.set_ylabel("Relative Density")
        axes.set_xlim(x_min, x_max)
        axes.hist(
            column,
            bins="fd",
            range=x_limits,
            density=True,
            label=f"{column.name} Value Distribution",
        )
        x = linspace(x_min, x_max, 2048)
        distribution = getattr(distributions, distribution_name)
        distribution_y = distribution.pdf(x, **distribution_parameters)
        axes.plot(x, distribution_y, label=f"Best Fit Model Distribution {distribution_name}")
        axes.legend()
    return reference


def _get_single_row(row_name: str | Hashable, data_frame: DataFrame) -> Series:
    """This is mostly a convenience wrapper due to poor pandas stubs."""
    return data_frame.loc[str(row_name)]  # type: ignore


async def _get_correlation_matrix(
    ctx: TaskContext,
    columns: DataFrame,
    fields: DataFrame,
) -> Optional[DataFrame]:
    filtered_column_names = fields.loc[fields[COMMON_UNIQUE] > 1].index

    if len(filtered_column_names) < 2:
        return None

    filtered_columns = columns[filtered_column_names]
    ctx.logger.debug("Computing correlation between columns %s", filtered_columns.columns)
    return filtered_columns.corr()


async def _get_correlation_graph(
    ctx: TaskContext, plot_name: PurePosixPath, correlation: Optional[DataFrame]
) -> Optional[FileReference]:
    if correlation is None:
        return None

    mask = triu(ones_like(correlation, dtype=bool))

    async with get_pyplot_writer(ctx, plot_name) as (axes, reference):
        figure = axes.figure
        if isinstance(figure, Figure):
            width_offset = 3
            height_offset = 3
            figure_size = (
                correlation.shape[0] + width_offset,
                correlation.shape[1] + height_offset,
            )
            figure.set_size_inches(figure_size)
        heatmap(
            correlation,
            annot=True,
            mask=mask,
            fmt=".2f",
            vmin=-1.0,
            center=0.0,
            vmax=1.0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
            cmap=get_cmap(),
            ax=axes,
        )
    ctx.logger.debug("Finished computing correlation graph")
    return reference


async def _get_correlation_summary(ctx: TaskContext, correlation: Optional[DataFrame]) -> CorrelationSummary:
    summary = CorrelationSummary()
    if correlation is None:
        return summary

    for col1, col2 in itertools.combinations(correlation.columns, 2):
        corr = abs(cast(float, correlation.loc[col1, col2]))
        if corr < 0.5:
            summary.no += 1
        elif corr < 0.8:
            summary.partial += 1
        else:
            summary.strong += 1

    ctx.logger.debug("Finished computing correlation summary")
    return summary


async def _get_string_distribution_graph(ctx: TaskContext, column: Series) -> Optional[FileReference]:
    config = ctx.config.structured_config.distribution
    number_unique = column.nunique()
    if number_unique < config.minimum_number_unique_string:
        ctx.logger.debug(
            'Skipping string distribution of column "%s", having only %d out of %d required unique values',
            column.name,
            number_unique,
            config.minimum_number_unique_string,
        )
        return None

    uniques = column.value_counts()
    uniques.sort_values(ascending=False)
    if len(uniques.index) > 20:
        uniques = uniques[:20]

    # Anonymize index
    uniques.index = RangeIndex(start=0, stop=len(uniques.index))
    plot_name = ctx.build_output_reference(f"{column.name}_distribution")
    async with get_pyplot_writer(ctx, plot_name) as (axes, reference):
        uniques.plot(ax=axes, kind="bar", title=f"Abundance of values in {column.name}")
    return reference
