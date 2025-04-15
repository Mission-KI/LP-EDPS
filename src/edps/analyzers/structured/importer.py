import asyncio
import csv
import io
from csv import Dialect
from io import TextIOBase
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional
from warnings import warn

from clevercsv import Detector, field_size_limit
from clevercsv.cparser import Parser
from clevercsv.detect import DetectionMethod
from clevercsv.dialect import SimpleDialect
from clevercsv.encoding import get_encoding
from clevercsv.potential_dialects import get_dialects as _get_dialects
from extended_dataset_profile import StructuredDataSet
from pandas import DataFrame, read_csv, read_excel

from edps.taskcontext import TaskContext

from ..structured import PandasAnalyzer


class HeaderMismatchWarning(UserWarning):
    """
    Custom warning indicating a mismatch between the number of headers
    and the number of columns in the csv data.
    """


async def csv_importer(ctx: TaskContext, path: Path) -> StructuredDataSet:
    ctx.logger.info("Analyzing CSV '%s'", ctx.relative_path(path))
    data_frame = await csv_import_dataframe(ctx, path)
    return await PandasAnalyzer(data_frame).analyze(ctx)


async def pandas_importer(ctx: TaskContext, data_frame: DataFrame) -> StructuredDataSet:
    ctx.logger.info("Analyzing structured dataset '%s'", ctx.qualified_path)
    return await PandasAnalyzer(data_frame).analyze(ctx)


async def csv_import_dataframe(ctx: TaskContext, path: Path) -> DataFrame:
    encoding = _detect_encoding(ctx, path)
    top_lines = _read_file_top_lines(path, encoding, num_lines=100)
    dialect = _detect_dialect(top_lines)
    has_header = _detect_header(top_lines)
    csv_dialect = _translate_dialect(ctx, dialect)
    ctx.logger.info("%s", "Detected Header" if has_header else "No Header detected")

    header = _read_header(top_lines, csv_dialect) if has_header else []
    data_frame: DataFrame = DataFrame.from_records(CSVParser(path, header, encoding, csv_dialect))

    column_count = len(data_frame.columns)
    header_mismatch = has_header and column_count != len(header)

    if header_mismatch:
        message = f"The header count {len(header)} does not match number of columns {column_count}"
        warn(message, HeaderMismatchWarning)
        ctx.logger.warning(message)

    return data_frame


def _translate_dialect(ctx, dialect: None | SimpleDialect) -> None | Dialect:
    if dialect:
        converted_dialect = dialect.to_csv_dialect()
        ctx.logger.info("Detected dialect: %s", dialect_to_str(converted_dialect))
        return converted_dialect
    else:
        ctx.logger.info("No dialect detected")
        return None


async def excel_importer(ctx: TaskContext, path: Path, engine: Literal["xlrd", "openpyxl"]) -> StructuredDataSet:
    """Import XLS/XLSX files. The engine must be passed explicitly to ensure the required libraries are installed."""

    ctx.logger.info("Analyzing Excel file '%s'", ctx.relative_path(path))
    dataframes_map = await excel_import_dataframes(ctx, path, engine)

    sheets = list(dataframes_map.keys())
    dataframes = list(dataframes_map.values())

    # TODO: Importer should be able to provide multiple outputs (DataFrames) in the future!
    if len(sheets) == 0:
        raise RuntimeError("Excel contains no sheets!")
    if len(sheets) == 1:
        ctx.logger.info("Data imported from sheet '%s'", sheets[0])
    if len(sheets) > 1:
        ctx.logger.warning("Excel contains multiple sheets %s, only first one is used!", sheets)

    return await PandasAnalyzer(dataframes[0]).analyze(ctx)


async def excel_import_dataframes(
    ctx: TaskContext, path: Path, engine: Literal["xlrd", "openpyxl"]
) -> dict[str, DataFrame]:
    """Import XLS/XLSX files. The engine must be passed explicitly to ensure the required libraries are installed."""

    return await asyncio.to_thread(
        lambda: read_excel(
            path,
            engine=engine,
            sheet_name=None,  # imports all sheets as dictionary
            header=0,  # assume header in row 0
        )
    )


async def xlsx_importer(ctx: TaskContext, path: Path) -> StructuredDataSet:
    return await excel_importer(ctx, path, "openpyxl")


async def xls_importer(ctx: TaskContext, path: Path) -> StructuredDataSet:
    return await excel_importer(ctx, path, "xlrd")


def get_possible_csv_dialects(opened_file: TextIOBase):
    data = opened_file.read()
    encoding = opened_file.encoding or "utf-8"
    return _get_dialects(data, encoding=encoding)


def dialect_to_str(dialect: Dialect) -> str:
    text = ""
    text += f'delimiter "{dialect.delimiter}", ' if dialect.delimiter else "no delimiter, "
    text += f'escapechar "{dialect.escapechar}", ' if dialect.escapechar else "no escapechar, "
    text += f'quotechar "{dialect.quotechar}"' if dialect.quotechar else "no quotechar"
    return text


class CSVParser:
    def __init__(self, filename: Path, header: List[str], encoding: str, dialect: Optional[csv.Dialect] = None) -> None:
        self._file = open(filename, "rt", newline="", encoding=encoding)
        self._header = header
        self._dialect = dialect or csv.Dialect()
        self._parser: Iterator[List[str]] = Parser(
            self._file,
            delimiter=self._dialect.delimiter,
            quotechar=self._dialect.quotechar,
            escapechar=self._dialect.escapechar,
            field_limit=field_size_limit(),
            strict=False,
            return_quoted=False,
        )
        # If header values are provided, skip the first row.
        if any(header):
            next(self._parser, None)

    def __iter__(self) -> "CSVParser":
        return self

    def __next__(self) -> Dict[str, Any]:
        try:
            row = next(self._parser)
        except StopIteration:
            self._file.close()
            raise

        # Strip whitespace from each field in the row.
        row = [value.strip() for value in row]

        if len(row) > len(self._header):
            self._header = self._header + [f"col{i:03d}" for i in range(len(self._header), len(row))]

        return dict(zip(self._header, row))


def _detect_encoding(ctx: TaskContext, path: Path) -> str:
    enc = get_encoding(path)
    if enc is None:
        raise RuntimeError(f'Could not detect encoding for "{ctx.relative_path(path)}"')
    ctx.logger.info('Detected encoding "%s"', enc)
    return enc


def _detect_dialect(data: str) -> SimpleDialect | None:
    detector = Detector()
    dialect = detector.detect(data, verbose=False, method=DetectionMethod.AUTO)
    return dialect


def _detect_header(data: str) -> bool:
    detector = Detector()
    has_header = detector.has_header(data)
    return has_header


def _read_header(data: str, dialect: Dialect | None) -> List[str]:
    buffer = io.StringIO(data)
    header_df = read_csv(
        buffer,
        header="infer",
        dialect=dialect,  # type: ignore
        nrows=1,
    )
    return header_df.columns.tolist()


def _read_file_top_lines(path: Path, enc: str, num_lines: int | None) -> str:
    data: str = ""
    # newline="" is important for CSV files as it preserves line endings.
    # Thus read() passes them through to the detector.
    with path.open("r", newline="", encoding=enc) as fp:
        for index, line in enumerate(fp):
            if not num_lines or index < num_lines:
                data += line
            else:
                break
    return data
