import json
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Tuple, Union

from extended_dataset_profile.models.v0.edp import SemiStructuredDataSet
from genson import SchemaBuilder
from pandas import DataFrame, json_normalize

from edps.taskcontext import TaskContext

from ..structured.importer import pandas_importer

JSONData = Union[Dict[str, Any], List[Any]]
StructuredDataMatch = Tuple[Dict[str, Any], str]


class JsonAnalyzer:
    def __init__(self, json_data: JSONData):
        self._json_data = json_data

    async def analyze(self, ctx: TaskContext) -> SemiStructuredDataSet:
        builder = SchemaBuilder()
        builder.add_object(self._json_data)
        schema = builder.to_schema()
        schema_string = json.dumps(schema, indent=2)
        ctx.logger.debug("Detected JSON schema:\n%s", schema_string)

        matches = list(_detect_structured_data(schema))

        num_dataframes = 0
        async for dataframe in self._extract_dataframes(ctx, matches):
            num_dataframes += 1
            await ctx.exec(f"dataframe_{num_dataframes:03}", pandas_importer, dataframe)

        return SemiStructuredDataSet(
            jsonSchema=schema_string,
        )

    async def _extract_dataframes(
        self, ctx: TaskContext, matches: List[StructuredDataMatch]
    ) -> AsyncIterator[DataFrame]:
        ctx.logger.info("Extracting dataframes...")
        for _, path in matches:
            structured_data = list(_extract_structured_data(self._json_data, path))
            dataframe = json_normalize(list(structured_data))
            yield dataframe


def _detect_structured_data(schema_node: Dict[str, Any], path: str = "") -> Iterable[StructuredDataMatch]:
    json_type = _get_json_type(schema_node)
    if json_type == "object":
        for prop_name, prop_schema in schema_node.get("properties", {}).items():
            child_path = f"{path}.{prop_name}" if path else prop_name
            yield from _detect_structured_data(prop_schema, child_path)
    elif json_type == "array":
        # The items type cannot always be detected, e.g. if there is an empty list in the input data.
        items_schema: Optional[dict] = schema_node.get("items")
        if items_schema and _get_json_type(items_schema) == "object":
            yield items_schema, path
            yield from _detect_structured_data(items_schema, path)


def _extract_structured_data(json_data: JSONData, path: str | List[str]) -> Iterable[Any]:
    if isinstance(path, str):
        if path:
            path = path.split(".")
        else:
            path = []

    if not path:
        if not (isinstance(json_data, list) and all(isinstance(item, dict) for item in json_data)):
            raise ValueError(
                f"Invalid path: expected a list of dicts (structured data), but got '{type(json_data).__name__}'."
            )
        yield from json_data
    else:
        if isinstance(json_data, dict):
            key, *remaining_parts = path
            if key not in json_data:
                raise ValueError(f"Invalid path: expected a dict containing key '{key}'.")
            yield from _extract_structured_data(json_data[key], remaining_parts)
        elif isinstance(json_data, list):
            for item in json_data:
                yield from _extract_structured_data(item, path)
        else:
            raise ValueError(
                f"Invalid path: expected a dict or list when processing path '{path}', but got '{type(json_data).__name__}'."
            )


def _get_json_type(schema_node: Dict[str, Any]) -> Optional[str]:
    json_type = schema_node.get("type")
    if json_type is None:
        # There is not always a "type" field but maybe an "anyOf" field. We don't support this inhomogeneous case.
        return None
    elif isinstance(json_type, list):
        # Union of primitive types, e.g. ["null", "string"]. This is not relevant as we only care about "array" and "object".
        return None
    elif isinstance(json_type, str):
        json_type_str = json_type.lower().strip()
        # Supported types in JSON schema: https://json-schema.org/understanding-json-schema/reference/type
        if json_type_str not in ["array", "boolean", "integer", "null", "number", "object", "string"]:
            raise RuntimeError(f"Found an invalid type in the JSON schema: '{json_type_str}'")
        return json_type_str
    else:
        raise ValueError(f"Found an invalid type in the JSON schema: '{json_type}'")
