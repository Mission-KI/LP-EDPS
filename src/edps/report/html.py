import os
from dataclasses import asdict
from datetime import datetime
from io import BufferedIOBase
from pathlib import Path
from typing import Any, List, Optional, Type, cast

from extended_dataset_profile import (
    ArchiveDataSet,
    AudioDataSet,
    DatasetTreeNode,
    DateTimeColumn,
    DocumentDataSet,
    ExtendedDatasetProfile,
    ImageDataSet,
    JsonReference,
    NumericColumn,
    SemiStructuredDataSet,
    StringColumn,
    StructuredDataSet,
    UnstructuredTextDataSet,
    VideoDataSet,
)
from jinja2 import Environment, PackageLoader, Undefined
from jinja2.exceptions import TemplateRuntimeError, UndefinedError
from jinja2.utils import missing
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

from ..report.base import ReportGenerator, ReportInput
from ..taskcontext import TaskContext
from ..types import DataSet


class HtmlReportGenerator(ReportGenerator):
    """Generates a HTML report."""

    def __init__(self):
        self._env = _init_environment()

    async def generate(self, ctx: TaskContext, input: ReportInput, base_dir: Path, output_buffer: BufferedIOBase):
        template = self._env.get_template("report.html.jinja")
        transformed_data = prepare_data(input)
        html = template.render(asdict(transformed_data))
        output_buffer.write(html.encode("utf-8"))
        self._check_output(ctx, output_buffer)

    def _check_output(self, ctx: TaskContext, output_buffer: BufferedIOBase):
        output_buffer.seek(0, os.SEEK_END)
        size = output_buffer.tell()
        if size == 0:
            raise RuntimeError("HTML report is empty.")
        ctx.logger.info("HTML report generated (%d bytes).", size)


def _init_environment():
    env = Environment(
        extensions=["jinja2.ext.debug", "jinja2.ext.do"],
        loader=PackageLoader("edps.report"),
        autoescape=True,
        undefined=PatchedUndefined,
    )

    env.tests["type_ArchiveDataSet"] = lambda val: isinstance(val, ArchiveDataSet)
    env.tests["type_StructuredDataset"] = lambda val: isinstance(val, StructuredDataSet)
    env.tests["type_SemiStructuredDataSet"] = lambda val: isinstance(val, SemiStructuredDataSet)
    env.tests["type_UnstructuredTextDataSet"] = lambda val: isinstance(val, UnstructuredTextDataSet)
    env.tests["type_DocumentDataSet"] = lambda val: isinstance(val, DocumentDataSet)
    env.tests["type_ImageDataSet"] = lambda val: isinstance(val, ImageDataSet)
    env.tests["type_VideoDataSet"] = lambda val: isinstance(val, VideoDataSet)
    env.tests["type_AudioDataSet"] = lambda val: isinstance(val, AudioDataSet)

    env.tests["type_NumericColumn"] = lambda val: isinstance(val, NumericColumn)
    env.tests["type_StringColumn"] = lambda val: isinstance(val, StringColumn)
    env.tests["type_DateTimeColumn"] = lambda val: isinstance(val, DateTimeColumn)
    return env


@dataclass
class DatasetWrapper:
    node: DatasetTreeNode
    details: DataSet
    qualifiedName: str


@dataclass
class ReportData(ReportInput):
    datasets: List[DatasetWrapper] = Field(default_factory=list)
    currentDate: datetime = Field(default_factory=datetime.now)


def prepare_data(input: ReportInput) -> ReportData:
    data = ReportData(edp=input.edp)

    for node in input.edp.datasetTree:
        details = _resolve_json_ref(input.edp, node.dataset)
        qualified_name = _build_qualified_name(input.edp, node)
        data.datasets.append(DatasetWrapper(node, details, qualified_name))

    return data


def _resolve_json_ref(edp: ExtendedDatasetProfile, json_ref: Optional[JsonReference]) -> Any:
    if not json_ref:
        return None

    segments = json_ref.reference.split("/")
    if segments[0] != "#":
        raise ValueError(f"JSON reference needs to be document-relative and start with '#/': {json_ref.reference}")
    obj: Any = edp
    for i, segment in enumerate(segments[1:]):
        try:
            if segment.isdigit():
                obj = obj[int(segment)]
            else:
                obj = getattr(obj, segment)
        except Exception as exception:
            raise ValueError(f"Can't resolve JSON reference {json_ref.reference}. Error on segment {i}: {exception}")
    return obj


def _resolve_json_ref_typed[T: BaseModel](
    edp: ExtendedDatasetProfile, json_ref: Optional[JsonReference], expected_type: type[T]
) -> Optional[T]:
    obj = _resolve_json_ref(edp, json_ref)
    return cast(T, obj) if obj else None


def _build_qualified_name(edp: ExtendedDatasetProfile, node: DatasetTreeNode) -> str:
    parent = _resolve_json_ref_typed(edp, node.parent, DatasetTreeNode)
    if not parent:
        return node.name
    else:
        parent_name = _build_qualified_name(edp, parent)
        return f"{parent_name}/{node.name}"


# We need to patch Jinja's "Undefined" class and pass it to the environment
# because we want to detect errors in the Jinja template navigation path.
class PatchedUndefined(Undefined):
    def __init__(
        self,
        hint: Optional[str] = None,
        obj: Any = missing,
        name: Optional[str] = None,
        exc: Type[TemplateRuntimeError] = UndefinedError,
    ) -> None:
        raise ValueError(f"Jinja template refers to undefined attribute '{name}'")

    def __getattr__(self, name: str):
        raise AttributeError()

    def __getitem__(self, *args: Any, **kwargs: Any):
        raise AttributeError()
