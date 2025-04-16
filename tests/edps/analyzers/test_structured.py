from pathlib import Path, PurePosixPath

import pytest
from extended_dataset_profile import CorrelationSummary
from pandas import DataFrame

from edps.analyzers.structured import (
    COMMON_UNIQUE,
    _get_correlation_graph,
    _get_correlation_matrix,
    _get_correlation_summary,
)
from edps.analyzers.structured.importer import (
    HeaderMismatchWarning,
    csv_import_dataframe,
    excel_import_dataframes,
    pandas_importer,
)


async def test_correlation(ctx):
    FIRST_COL = "elevator_accidents"
    SECOND_COL = "maintenance_interval"
    columns = DataFrame({FIRST_COL: [1, 2, 3, 4, 5], SECOND_COL: [5.0, 6.2, 7.5, 8.1, 9.9]})
    fields = DataFrame({COMMON_UNIQUE: [5, 5]}, index=[FIRST_COL, SECOND_COL])
    correlation_matrix = await _get_correlation_matrix(
        ctx,
        columns,
        fields,
    )
    correlation_graph = await _get_correlation_graph(ctx, PurePosixPath("test_correlation_graph"), correlation_matrix)
    correlation_summary = await _get_correlation_summary(ctx, correlation_matrix)

    assert correlation_graph is not None
    assert correlation_summary == CorrelationSummary(no=0, partial=0, strong=1)


async def test_import_csv(path_data_test_csv, ctx):
    data = await csv_import_dataframe(ctx, path_data_test_csv)
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 5
    assert row_count == 50
    assert headers == ["uuid", "einfahrt", "ausfahrt", "aufenthalt", "parkhaus"]


async def test_import_csv_no_headers(path_data_test_headerless_csv, ctx):
    data = await csv_import_dataframe(ctx, path_data_test_headerless_csv)
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 5
    assert row_count == 50
    assert headers == ["col000", "col001", "col002", "col003", "col004"]


async def test_import_csv_extra_headers(path_data_test_extra_headers_csv, ctx):
    with pytest.warns(HeaderMismatchWarning):
        data = await csv_import_dataframe(ctx, path_data_test_extra_headers_csv)
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 5
    assert row_count == 6
    assert headers == ["uuid", "einfahrt", "ausfahrt", "aufenthalt", "a1"]


async def test_import_csv_missing_headers(path_data_test_missing_headers_csv, ctx):
    with pytest.warns(HeaderMismatchWarning):
        data = await csv_import_dataframe(ctx, path_data_test_missing_headers_csv)
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 5
    assert row_count == 6
    assert headers == ["uuid", "einfahrt", "ausfahrt", "aufenthalt", "col004"]


async def test_import_csv_with_clevercsv(ctx, tmp_path: Path):
    """Test the advanced detection of clevercsv.

    The standard csv would fail with 'Could not determine delimiter'
    """

    csv_file = tmp_path / "clever.csv"
    csv_file.write_text(
        """Idx;1 col;;3 col;;5 col;
2022;28 ;16.8 ;49 ;5.7 ;78 ;4.5
2023;42 ;17.0 ;28 ;5.3 ;61 ;4.3
"""
    )
    data = await csv_import_dataframe(ctx, csv_file)
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert headers == ["Idx", "1 col", "Unnamed: 2", "3 col", "Unnamed: 4", "5 col", "Unnamed: 6"]
    assert col_count == 7
    assert row_count == 2
    first_row = data.iloc[0]
    assert first_row.to_list() == ["2022", "28", "16.8", "49", "5.7", "78", "4.5"]


async def test_import_csv_with_quoted_strings(ctx, tmp_path: Path):
    csv_file = tmp_path / "clever.csv"
    csv_file.write_text(
        """id;value;num
1;"hello";2023
2;"world";2024
"""
    )
    data = await csv_import_dataframe(ctx, csv_file)
    assert data.to_dict() == {
        "id": {0: "1", 1: "2"},  # TODO(KB) index needs to be fixed
        "value": {0: "hello", 1: "world"},
        "num": {0: "2023", 1: "2024"},
    }


async def test_import_xlsx(path_data_test_xlsx, ctx):
    dataframes_map = await excel_import_dataframes(ctx, path_data_test_xlsx, "openpyxl")
    data = list(dataframes_map.values())[0]
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 5
    assert row_count == 50
    assert headers == ["uuid", "einfahrt", "ausfahrt", "aufenthalt", "parkhaus"]


async def test_import_xls(path_data_test_xls, ctx):
    dataframes_map = await excel_import_dataframes(ctx, path_data_test_xls, "xlrd")
    data = list(dataframes_map.values())[0]
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 5
    assert row_count == 50
    assert headers == ["uuid", "einfahrt", "ausfahrt", "aufenthalt", "parkhaus"]


async def test_detect_german_decimal_comma(path_data_german_decimal_comma_csv, ctx):
    data = await csv_import_dataframe(ctx, path_data_german_decimal_comma_csv)
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 15
    assert row_count == 6
    assert headers == [
        "Zeit",
        "0-17 Jahre",
        "Unnamed: 2",
        "18-24 Jahre",
        "Unnamed: 4",
        "25-29 Jahre",
        "Unnamed: 6",
        "30-49 Jahre",
        "Unnamed: 8",
        "50-64 Jahre",
        "Unnamed: 10",
        "65 Jahre und aelter",
        "Unnamed: 12",
        "Insgesamt",
        "Unnamed: 14",
    ]


async def test_hamburg(path_data_hamburg_csv, ctx):
    with pytest.warns(HeaderMismatchWarning):
        data = await csv_import_dataframe(ctx, path_data_hamburg_csv)
    row_count = len(data.index)
    col_count = len(data.columns)
    headers = data.columns.tolist()
    assert col_count == 13
    assert row_count == 55
    assert headers == [
        "Unnamed: 0",
        "Unnamed: 1",
        "Unnamed: 2",
        "Unnamed: 3",
        "Unnamed: 4",
        "Unnamed: 5",
        "in Tsd. Euro",
        "col007",
        "col008",
        "col009",
        "col010",
        "col011",
        "col012",
    ]


async def test_analyse_dataframe(ctx):
    cells = [
        ["- ", "", "MAJOR", "", " breaking changes  "],
        ["- ", "", "MINOR", "", "   new features   "],
        ["- ", "", "PATCH", "", "bug fixes"],
    ]
    df = DataFrame(columns=cells[0], data=cells[1:])
    dataset = await pandas_importer(ctx, df)
    headers = [col.name for col in dataset.all_columns]
    assert dataset.columnCount == 5
    assert dataset.rowCount == 2
    assert headers == [
        "-",
        "col001",
        "MAJOR",
        "col003",
        "breaking changes",
    ]
