from pathlib import Path, PurePosixPath
from typing import AsyncIterator
from zipfile import ZipFile

from docx import Document as read_document
from docx.document import Document
from extended_dataset_profile import (
    DocumentDataSet,
    ImageDataSet,
    ModificationState,
)
from pandas import DataFrame

from edps.analyzers.common import split_keywords
from edps.analyzers.structured.importer import pandas_importer
from edps.analyzers.unstructured_text.importer import unstructured_text_importer_from_string
from edps.taskcontext import TaskContext


async def docx_importer(ctx: TaskContext, path: Path) -> DocumentDataSet:
    ctx.logger.info("Analyzing DOCX '%s'", ctx.relative_path(path))

    doc = read_document(str(path))
    return await DocxAnalyzer(doc, path).analyze(ctx)


class DocxAnalyzer:
    def __init__(self, doc: Document, path: Path):
        self.doc = doc
        self.path = path

    async def analyze(self, ctx: TaskContext) -> DocumentDataSet:
        props = self.doc.core_properties
        keywords = split_keywords(props.keywords)
        modified = props.revision > 1

        extracted_text = self._extract_text(ctx)
        await ctx.exec("text", unstructured_text_importer_from_string, extracted_text)

        num_tables = 0
        async for dataframe in self._extract_tables(ctx):
            num_tables += 1
            await ctx.exec(f"table_{num_tables:03}", pandas_importer, dataframe)

        async for media_file in self._extract_media_files(ctx):
            ctx.logger.info("Analyzing DOCX embedded media file '%s'...", ctx.relative_path(media_file))
            await ctx.import_file(media_file)

        num_images = len([ds for ds in ctx.collect_datasets() if isinstance(ds, ImageDataSet)])

        return DocumentDataSet(
            title=props.title,
            subject=props.subject,
            author=props.author,
            toolchain=None,
            creationDate=props.created,
            modificationDate=props.modified,
            keywords=keywords,
            docType="DOCX",
            numPages=None,  # unknown, this would need rendering the pages
            numImages=num_images,
            modified=ModificationState.modified if modified else ModificationState.unmodified,
            encrypted=False,
        )

    def _extract_text(self, ctx: TaskContext) -> str:
        ctx.logger.info("Extracting text...")
        paragraphs = self.doc.paragraphs
        paragraph_texts = [p.text for p in paragraphs]
        return "\n".join(paragraph_texts)

    async def _extract_media_files(self, ctx: TaskContext) -> AsyncIterator[Path]:
        ctx.logger.info("Extracting media files...")
        # Open the .docx file as a zip archive
        with ZipFile(self.path, "r") as docx_zip:
            # Filter files in the 'word/media/' folder (which contains images)
            file_list = docx_zip.namelist()
            media_files = [file for file in file_list if file.startswith("word/media/")]
            if not media_files:
                return

            unzipped_media_dir = ctx.create_working_dir("media")
            ctx.logger.debug("Unzipping %d media files...", len(media_files))
            for media_file in media_files:
                unzipped_media_file_path = Path(unzipped_media_dir) / PurePosixPath(media_file).name
                with unzipped_media_file_path.open("wb") as file_io:
                    # Unzip the image to a temp file
                    file_io.write(docx_zip.read(media_file))
                    ctx.logger.debug("Extracted '%s' from DOCX to '%s'", media_file, unzipped_media_file_path)
                yield unzipped_media_file_path

    async def _extract_tables(self, ctx: TaskContext) -> AsyncIterator[DataFrame]:
        ctx.logger.info("Extracting tables...")
        for table in self.doc.tables:
            cells: list[list[str]] = []
            for row in table.rows:
                row_cells = [c.text for c in row.cells]
                cells.append(row_cells)
            # Assume first row is header
            yield DataFrame(cells[1:], columns=cells[0])
