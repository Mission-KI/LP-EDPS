import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

from numpy import any as numpy_any
from pandas import DataFrame, Index, Series, StringDtype, to_datetime, to_numeric
from pydantic.dataclasses import dataclass

from edps.taskcontext import TaskContext


@dataclass(frozen=True)
class _BaseColumnInfo:
    number_nan_before_conversion: int
    number_inconsistent: int
    number_interpretable: int


class DatetimeKind(str, Enum):
    UNKNOWN = "UNKNOWN"
    DATETIME = "DATETIME"
    DATE = "DATE"
    TIME = "TIME"


@dataclass(frozen=True)
class DatetimeColumnInfo(_BaseColumnInfo):
    format: str
    kind: DatetimeKind


@dataclass(frozen=True)
class NumericColumnInfo(_BaseColumnInfo):
    pass


@dataclass(frozen=True)
class StringColumnInfo(_BaseColumnInfo):
    pass


ColumnInfo = Union[DatetimeColumnInfo, NumericColumnInfo, StringColumnInfo]


class _ColumnConverter(ABC):
    @abstractmethod
    def cast(self, ctx: TaskContext, col: Series) -> Tuple[Series, ColumnInfo]:
        """This shall convert the column to the destination type and return NA for entries it cannot parse."""

    def count_valid_rows(self, ctx: TaskContext, col: Series) -> int:
        """This counts how many entries in the column can be converted to the destination type."""
        converted_col, _ = self.cast(ctx, col)
        return converted_col.notna().sum()

    @abstractmethod
    def __str__(self) -> str:
        """Shall return a string describing the converter"""


class _StringColumnConverter(_ColumnConverter):
    def cast(self, ctx: TaskContext, col: Series) -> Tuple[Series, ColumnInfo]:
        # There is no nan or null when interpreting the column as string.
        # So we need to check for empty strings.
        converted = col.astype(StringDtype())
        number_nan = _count_empty(converted)
        interpretable = converted.count()
        return converted, StringColumnInfo(
            number_nan_before_conversion=number_nan, number_inconsistent=0, number_interpretable=interpretable
        )

    def __str__(self):
        return "StringColumnConverter"


class _DatetimeColumnConverter(_ColumnConverter):
    @abstractmethod
    def get_format(self) -> str: ...

    @abstractmethod
    def get_kind(self) -> DatetimeKind: ...

    @abstractmethod
    def _convert_entry(self, element: Any) -> None | datetime: ...

    def count_valid_rows(self, ctx: TaskContext, col: Series) -> int:
        converted_col = self._cast_internal(ctx, col)
        return converted_col.notna().sum()

    def cast(self, ctx: TaskContext, col: Series) -> Tuple[Series, ColumnInfo]:
        if self.get_kind() == DatetimeKind.TIME:
            _warning(ctx, f"Column '{col.name}' contains pure times without dates. This is handled as a string column.")
            return _StringColumnConverter().cast(ctx, col)

        number_nan = _count_empty(col)
        converted = self._cast_internal(ctx, col)
        number_inconsistent = converted.isna().sum() - number_nan
        number_interpretable = converted.count()

        if self.get_kind() == DatetimeKind.DATE:
            _warning(ctx, f"Column '{col.name}' contains pure dates without times.")
        return converted, DatetimeColumnInfo(
            number_nan_before_conversion=number_nan,
            number_inconsistent=number_inconsistent,
            number_interpretable=number_interpretable,
            format=self.get_format(),
            kind=self.get_kind(),
        )

    def _cast_internal(self, ctx: TaskContext, col: Series) -> Series:
        col_converted = col.apply(self._convert_entry)
        col_valid = col_converted[col_converted.notna()]
        count_with_timezone = col_valid.apply(lambda timestamp: timestamp.tzinfo is not None).sum()
        count_without_timezone = len(col_valid.index) - count_with_timezone

        if count_with_timezone == 0:
            # If there are no entries with timezone we return them as dtype "datetime64[ns]" without TZ info.
            return to_datetime(col_converted, errors="coerce", utc=False)
        else:
            if count_without_timezone > 0:
                _warning(
                    ctx,
                    f"Found datetime column '{col.name}' which contains entries partially with and without timezone. For entries without timezone UTC is assumed.",
                )
            # If there is at least one entry with a timezone we convert them to dtype "datetime64[ns, UTC]" with TZ=UTC.
            return to_datetime(col_converted, errors="coerce", utc=True)

    def __str__(self):
        return f"DatetimeColumnConverter(format='{self.get_format()}', kind={self.get_kind()})"


class _DatetimeNativeColumnConverter(_DatetimeColumnConverter):
    def get_format(self) -> str:
        return "NATIVE"

    def get_kind(self) -> DatetimeKind:
        return DatetimeKind.UNKNOWN

    def _convert_entry(self, element: Any) -> None | datetime:
        if isinstance(element, datetime):
            return element
        else:
            return None


class _DatetimeIsoColumnConverter(_DatetimeColumnConverter):
    def get_format(self) -> str:
        return "ISO8601"

    def get_kind(self) -> DatetimeKind:
        return DatetimeKind.UNKNOWN

    def _convert_entry(self, element: Any) -> None | datetime:
        try:
            return datetime.fromisoformat(element)
        except Exception:
            return None

    def __str__(str):
        return "DatetimeIsoConverter"


@dataclass(frozen=True)
class _DatetimePatternColumnConverter(_DatetimeColumnConverter):
    format: str
    kind: DatetimeKind

    def get_format(self) -> str:
        return self.format

    def get_kind(self) -> DatetimeKind:
        return self.kind

    def _convert_entry(self, element: Any) -> None | datetime:
        try:
            return datetime.strptime(element, self.format)
        except Exception:
            return None


class _NumericColumnConverter(_ColumnConverter):
    def count_valid_rows(self, ctx: TaskContext, col: Series) -> int:
        col_numeric = to_numeric(self._preprocess(col), errors="coerce")
        return col_numeric.notna().sum()

    def cast(self, ctx: TaskContext, col: Series) -> Tuple[Series, ColumnInfo]:
        number_nan = _count_empty(col)
        # Convert to numeric type and set incompatible values to float.nan.
        col_numeric = to_numeric(self._preprocess(col), errors="coerce")
        # Narrow type to one supporting pd.NA (Int64 or Float64).
        col_numeric_nullable = self._cast_to_pandas_type(col_numeric)
        # Downcast to most specific type (e.g. UInt8, Int16, Float32 etc.)
        converted = self._downcast(col_numeric_nullable)
        ctx.logger.debug("Casting column '%s' to numeric dtype %s", col.name, converted.dtype)
        number_inconsistent = converted.isna().sum() - number_nan
        number_interpretable = converted.count()
        return converted, NumericColumnInfo(
            number_nan_before_conversion=number_nan,
            number_inconsistent=number_inconsistent,
            number_interpretable=number_interpretable,
        )

    def _preprocess(self, col: Series) -> Series:
        # Try replacing decimal separator "," with ".".
        return col.apply(lambda val: val.replace(",", ".") if isinstance(val, str) else val)

    def _cast_to_pandas_type(self, col: Series) -> Series:
        # There is no Pandas type for complex numbers. This is okay as Numpy's complex NA.
        if col.dtype.kind in "c":
            return col
        # Convert to Pandas types because they support NA (e.g. Int8).
        # For Numpy NA is represented as float.nan, so downcasting is not possible!
        with warnings.catch_warnings():
            # convert_dtypes() raises a warning when trying to convert certain float values:
            # "RuntimeWarning: invalid value encountered in cast"
            # at pandas\core\dtypes\cast.py", line 1056, in convert_dtypes
            # This is not a problem because it checks if the converted values are equal to the original ones.
            warnings.simplefilter("ignore", RuntimeWarning)
            return col.convert_dtypes(
                infer_objects=False,
                convert_string=False,
                convert_integer=True,
                convert_boolean=False,
                convert_floating=True,
            )

    def _downcast(self, col: Series) -> Series:
        # After that dtype should be Int64 or Float64. Try downcasting.
        if col.dtype.kind in "f":
            return to_numeric(col, downcast="float", errors="coerce")
        if col.dtype.kind in "i" and numpy_any(col < 0):
            return to_numeric(col, downcast="signed", errors="coerce")
        if col.dtype.kind in ("i", "u"):
            return to_numeric(col, downcast="unsigned", errors="coerce")
        # E.g. complex numbers
        return col

    def __str__(self):
        return "NumericColumnConverter"


ColumnConverter = Union[_DatetimeColumnConverter, _NumericColumnConverter, _StringColumnConverter]


class ColumnsWrapper[TColumn: ColumnInfo]:
    def __init__(self, all_data: DataFrame, infos: dict[str, TColumn]):
        self._all_data = all_data
        self._infos = infos

    @property
    def ids(self) -> list[str]:
        return list(self._infos.keys())

    def get_info(self, id: str) -> TColumn:
        return self._infos[id]

    @property
    def data(self) -> DataFrame:
        return self._all_data.loc[:, self.ids]

    def get_col(self, id: str) -> Series:
        return self._all_data.loc[:, id]

    def __iter__(self) -> Iterator[Tuple[str, TColumn, Series]]:
        for column_name, info in self._infos.items():
            yield column_name, info, self._all_data.loc[:, column_name]


class Result:
    def __init__(self, data: DataFrame, infos: Dict[str, ColumnInfo]) -> None:
        self._data = data
        self._infos: Dict[str, ColumnInfo] = infos

    def set_index(self, index: Index):
        self._data.set_index(index, inplace=True)

    @property
    def all_cols(self) -> ColumnsWrapper[ColumnInfo]:
        return ColumnsWrapper(self._data, self._infos)

    @property
    def datetime_cols(self) -> ColumnsWrapper[DatetimeColumnInfo]:
        infos = {key: value for key, value in self._infos.items() if isinstance(value, DatetimeColumnInfo)}
        return ColumnsWrapper(self._data, infos)

    @property
    def numeric_cols(self) -> ColumnsWrapper[NumericColumnInfo]:
        infos = {key: value for key, value in self._infos.items() if isinstance(value, NumericColumnInfo)}
        return ColumnsWrapper(self._data, infos)

    @property
    def string_cols(self) -> ColumnsWrapper[StringColumnInfo]:
        infos = {key: value for key, value in self._infos.items() if isinstance(value, StringColumnInfo)}
        return ColumnsWrapper(self._data, infos)


# First try datetime parsing (because "20240103" is an ISO date and a number):
# 1) Parse ISO datetimes with or without timezone, e.g. "20450630T13:29:53" or "20450630T13:29:53+0100".
#    If at least one entry has a TZ all are converted to dtype "datetime64[ns, UTC]" with TZ=UTC. For entries without timezone UTC is assumed.
#    If none has a timezone they are returned as dtype "datetime64[ns]" without timezone info.
# 2) Try parsing with different local formats (no time zone for now).
# Then try parsing as a number ("." and "," as decimal separator).
ALLOWED_COLUMN_CONVERTERS: List[_ColumnConverter] = [
    # Column already has datetime type (e.g. from Excel import)
    _DatetimeNativeColumnConverter(),
    # ISO datetime
    _DatetimeIsoColumnConverter(),
    # DE datetime
    _DatetimePatternColumnConverter(format="%d.%m.%Y %H:%M:%S", kind=DatetimeKind.DATETIME),
    _DatetimePatternColumnConverter(format="%d.%m.%Y %H:%M", kind=DatetimeKind.DATETIME),
    _DatetimePatternColumnConverter(format="%d.%m.%Y", kind=DatetimeKind.DATE),
    # US datetime
    _DatetimePatternColumnConverter(format="%m/%d/%Y %H:%M:%S", kind=DatetimeKind.DATETIME),
    _DatetimePatternColumnConverter(format="%m-%d-%Y %H:%M:%S", kind=DatetimeKind.DATETIME),
    _DatetimePatternColumnConverter(format="%m/%d/%Y %H:%M", kind=DatetimeKind.DATETIME),
    _DatetimePatternColumnConverter(format="%m-%d-%Y %H:%M", kind=DatetimeKind.DATETIME),
    _DatetimePatternColumnConverter(format="%m/%d/%Y", kind=DatetimeKind.DATE),
    _DatetimePatternColumnConverter(format="%m-%d-%Y", kind=DatetimeKind.DATE),
    # Time only
    _DatetimePatternColumnConverter(format="%H:%M:%S", kind=DatetimeKind.TIME),
    _DatetimePatternColumnConverter(format="%H:%M", kind=DatetimeKind.TIME),
    # Numeric
    _NumericColumnConverter(),
]
# String conversion as fallback because that is always possible.
FALLBACK_COLUMN_CONVERTER = _StringColumnConverter()
# Required minimum successful conversion ratio considering non-null values.
MIN_CONVERSION_RATIO = 0.5
# Conversion is first tried with a sample
SAMPLE_SIZE = 1000


def parse_types(ctx: TaskContext, data: DataFrame) -> Result:
    column_infos = dict[str, ColumnInfo]()
    results_data = DataFrame(index=data.index, columns=data.columns)

    if len(data.columns) != len(set([col for col in data.columns if col])):
        raise RuntimeError(f"The dataframe has empty or duplicate columns: {data.columns}")

    for column_name, column in data.items():
        if not isinstance(column_name, str):
            raise RuntimeError(f"Column {column_name} needs a label.")
        converted_column, col_info = _determine_type(ctx, column)
        column_infos[column_name] = col_info
        results_data[column_name] = converted_column

    result = Result(results_data, column_infos)

    ctx.logger.debug("Found Numeric columns: %s", result.numeric_cols.ids)
    ctx.logger.debug("Found DateTime columns: %s", result.datetime_cols.ids)
    ctx.logger.debug("Found String columns: %s", result.string_cols.ids)

    return result


def _determine_type(ctx: TaskContext, column: Series) -> tuple[Series, ColumnInfo]:
    non_empty_column = column[column.notna() & (column != "")]
    column_sample = (
        non_empty_column.sample(SAMPLE_SIZE) if len(non_empty_column.index) > SAMPLE_SIZE else non_empty_column
    )
    # determine best type match based on the sample
    preferred_col_converter = _determine_best_type_match(ctx, column_sample)
    new_column, column_info = _do_type_conversion(ctx, column, preferred_col_converter)
    _validate_selected_column_info(ctx, column, column_info)
    return new_column, column_info


def _determine_best_type_match(ctx: TaskContext, column: Series) -> Optional[_ColumnConverter]:
    # match: matched_colinfo, valid_count
    matches = Series({converter: count for converter, count in _try_type_conversions(ctx, column)}, name="valid-rows")
    non_null_matches = matches[matches.notnull()]
    if len(non_null_matches) > 0:
        return cast(_ColumnConverter, non_null_matches.idxmax())
    else:
        return None


def _try_type_conversions(ctx: TaskContext, column: Series) -> Iterator[tuple[_ColumnConverter, Optional[int]]]:
    # passed columns don't contain NA values anymore
    count = len(column.index)
    for converter in ALLOWED_COLUMN_CONVERTERS:
        valid_count = converter.count_valid_rows(ctx, column)
        ctx.logger.debug(
            "Trying to convert column '%s' with %s, found %d valid rows out of %d.",
            column.name,
            converter,
            valid_count,
            count,
        )
        # succesfully converted values must surpass min ratio
        if _check_conversion_ratio_passed(valid_count, count):
            yield converter, valid_count
        else:
            yield converter, None


def _do_type_conversion(
    ctx: TaskContext, column: Series, column_converter: Optional[_ColumnConverter]
) -> tuple[Series, ColumnInfo]:
    if column_converter is not None:
        # convert the whole column based on the preferred column info
        converted_col, column_info = column_converter.cast(ctx, column)

        if column_info.number_nan_before_conversion > 0:
            _warning(
                ctx,
                f"Column '{column.name}' contains {column_info.number_nan_before_conversion} empty of {len(column.index)} overall entries.",
            )
        if column_info.number_inconsistent > 0:
            _warning(
                ctx,
                f"Couldn't convert some entries of column '{column.name}' using {column_converter} ({column_info.number_inconsistent} invalid of {len(column.index)} overall entries).",
            )
        else:
            ctx.logger.debug(
                "Converting all %d non-null entries of column '%s' using %s.",
                column_info.number_interpretable,
                column.name,
                column_converter,
            )
        return converted_col, column_info

    # string column as fallback
    ctx.logger.debug("Converting column '%s' using FALLBACK %s.", column.name, FALLBACK_COLUMN_CONVERTER)
    return FALLBACK_COLUMN_CONVERTER.cast(ctx, column)


def _check_conversion_ratio_passed(valid_count: int, count: int):
    if count == 0:
        return False
    return valid_count >= MIN_CONVERSION_RATIO * count


DATETIME_MAGIC_WORDS = [
    "jahr",
    "monat",
    "tag",
    "stunde",
    "minute",
    "sekunde",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
]


def _validate_selected_column_info(ctx: TaskContext, col: Series, col_info: Optional[ColumnInfo]):
    col_label = f"{col.name}".lower().strip()
    if col_label in DATETIME_MAGIC_WORDS and not isinstance(col_info, DatetimeColumnInfo):
        _warning(
            ctx,
            f"The column '{col.name}' is maybe intended as a date/time column but the content could not be parsed. Please use ISO datetime format!",
        )


# TODO Find a better integrated solution
def _warning(ctx: TaskContext, message: str):
    ctx.logger.warning(message)
    warnings.warn(message)


def _count_empty(col: Series):
    return col.isna().sum() + (col == "").sum()
