import warnings
from logging import getLogger
from pathlib import Path

import pytest
from pypdf import PdfReader
from pytest import fixture, mark

from edps.report import HtmlReportGenerator, PdfReportGenerator, ReportInput
from edps.service import analyze_asset
from edps.taskcontext import TaskContext


@fixture
def report_output_path(path_work):
    path: Path = path_work / "report"
    path.mkdir(parents=True, exist_ok=True)
    return path


# This test generates multiple reports because fixture "asset_path" iterates through multiple assets.
@mark.slow
async def test_all_reports(ctx: TaskContext, asset_path, report_output_path, user_provided_data):
    # Ignore warnings which occur for some of the assets.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = await analyze_asset(asset_path, ctx, user_provided_data)

    await result.write_edp_to_output()
    # A PDF report should already have been created during normal asset analysis.
    # In this test we explicitly create another HTML and PDF report anyways.
    report_input = ReportInput(edp=result.edp)

    report_html = report_output_path / f"{asset_path.name}.html"
    with report_html.open("wb") as file_io:
        await HtmlReportGenerator().generate(ctx, report_input, file_io)
    assert report_html.exists()

    report_pdf = report_output_path / f"{asset_path.name}.pdf"
    with report_pdf.open("wb") as file_io:
        await PdfReportGenerator().generate(ctx, report_input, file_io)
    assert report_pdf.exists()

    pdf_text = read_pdf_text(report_pdf)
    assert "Title\ndataset-dummy-name" in pdf_text
    # Test if umlauts were encoded correctly
    assert "Tags\ntest, csv, Äöüß" in pdf_text

    getLogger().info("Report output path: %s", report_output_path)


def read_pdf_text(report_pdf: Path) -> str:
    with PdfReader(report_pdf) as pdf_reader:
        pages = pdf_reader.pages
        page_text = [p.extract_text(extraction_mode="plain") for p in pages]
        return "\n\n".join(page_text)


def test_chardet_dependency():
    """We are testing if we get a meaningful error message if we try to use the only function using "chardet".
    This is patched by edps.report.pdf.patch_chardet_dependency()."""

    from reportlab.lib.rparsexml import smartDecode  # type: ignore[import-untyped]

    with pytest.raises(RuntimeError) as exc_info:
        smartDecode("abc")
    assert (
        str(exc_info.value)
        == "We have deliberately excluded the 'chardet' dependency of 'reportlab' for license reasons."
    )
