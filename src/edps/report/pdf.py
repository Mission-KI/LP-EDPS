import os
from io import BufferedIOBase, BytesIO
from typing import Any

from xhtml2pdf import pisa
from xhtml2pdf.context import pisaContext as PisaContext

from ..file import build_real_sub_path
from ..report.base import ReportGenerator, ReportInput
from ..report.html import HtmlReportGenerator
from ..taskcontext import TaskContext


class PdfReportGenerator(ReportGenerator):
    """Generates a PDF report."""

    async def generate(self, ctx: TaskContext, input: ReportInput, output_buffer: BufferedIOBase):
        html_buffer = BytesIO()
        await HtmlReportGenerator().generate(ctx, input, html_buffer)

        pisa_context: PisaContext = pisa.CreatePDF(
            html_buffer,
            encoding="utf-8",
            dest=output_buffer,
            link_callback=lambda url, origin: build_real_sub_path(ctx.output_path, str(url)).as_posix(),
        )
        self._check_output(ctx, pisa_context, output_buffer)

    def _check_output(self, ctx: TaskContext, pisa_context: PisaContext, output_buffer: BufferedIOBase):
        # Print warnings and errors
        if len(pisa_context.log) > 0:
            ctx.logger.info("There were messages during PDF report generation:")
        for log_entry in pisa_context.log:
            level, _, msg, _ = log_entry
            if level == "error":
                ctx.logger.error(msg)
            elif level == "warning":
                ctx.logger.warning(msg)
            else:
                ctx.logger.info(msg)
        # Check warning and error counters
        if pisa_context.warn > 0:
            ctx.logger.warning("There were %d warnings during PDF report generation.", pisa_context.warn)
        if pisa_context.err > 0:
            ctx.logger.error("There were %d errors during PDF report generation.", pisa_context.err)

        output_buffer.seek(0, os.SEEK_END)
        size = output_buffer.tell()
        if size == 0:
            raise RuntimeError("PDF report is empty.")
        ctx.logger.info("PDF report generated (%d bytes).", size)


def patch_chardet_dependency():
    """
    For license reasons we have removed the "chardet" library which is a dependency of "reportlab" which is a dependency of "xhtml2pdf.
    For our use-case PDF generation works fine without chardet.
    We patch the only function using chardet to get a meaningful error message.
    """

    import reportlab.lib.rparsexml  # type: ignore[import-untyped]

    def fake_smart_decode(s: Any):
        raise RuntimeError("We have deliberately excluded the 'chardet' dependency of 'reportlab' for license reasons.")

    reportlab.lib.rparsexml.smartDecode = fake_smart_decode
