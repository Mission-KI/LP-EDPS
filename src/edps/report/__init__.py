from edps.report.base import ReportGenerator as ReportGenerator
from edps.report.base import ReportInput as ReportInput
from edps.report.html import HtmlReportGenerator as HtmlReportGenerator
from edps.report.pdf import PdfReportGenerator as PdfReportGenerator
from edps.report.pdf import patch_chardet_dependency

patch_chardet_dependency()
