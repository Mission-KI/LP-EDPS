import os
from io import BufferedIOBase, BytesIO
from pathlib import Path

from xhtml2pdf import pisa
from xhtml2pdf.context import pisaContext as PisaContext

from edps.file import build_real_sub_path
from edps.report.base import ReportGenerator, ReportInput
from edps.report.html import HtmlReportGenerator
from edps.taskcontext import TaskContext


class PdfReportGenerator(ReportGenerator):
    """Generates a PDF report."""

    async def generate(self, ctx: TaskContext, input: ReportInput, base_dir: Path, output_buffer: BufferedIOBase):
        html_buffer = BytesIO()
        await HtmlReportGenerator().generate(ctx, input, base_dir, html_buffer)

        pisa_context: PisaContext = pisa.CreatePDF(
            html_buffer,
            dest=output_buffer,
            link_callback=lambda url, origin: build_real_sub_path(base_dir, str(url)).as_posix(),
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
