"""
Generate a JSON schema for the FitProblem model.
Note:
This script is likely deprecated, and relies on pydantic v1,
as it makes use of functionality that have been removed in v2.
"""

import os
import math
import json

os.environ["BUMPS_USE_PYDANTIC"] = "True"

from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import PydanticOmit, core_schema
from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic import TypeAdapter


class CustomsGenerateJsonSchema(GenerateJsonSchema):
    def handle_invalid_for_json_schema(self, schema: core_schema.CoreSchema, error_info: str) -> JsonSchemaValue:
        print(f"Handling invalid schema: {error_info}")
        raise PydanticOmit

    def generate(self, schema: core_schema.CoreSchema, mode: str = "validation") -> JsonSchemaValue:
        json_schema = super().generate(schema, mode=mode)
        # Filter out properties starting with underscore
        filtered_properties = {k: v for k, v in json_schema.get("properties", {}).items() if not k.startswith("_")}
        json_schema["properties"] = filtered_properties
        filtered_requires = [k for k in json_schema.get("required", []) if not k.startswith("_")]
        json_schema["required"] = filtered_requires
        return json_schema

    def handle_property(self, property_name: str, property_value: JsonSchemaValue):
        if property_name.startswith("_"):
            return JsonSchemaValue(None)
        return super().handle_property(property_name, property_value)


from bumps.parameter import Expression, Parameter  # , UnaryExpression

from refl1d.bumps_interface.fitproblem import FitProblem

ta = TypeAdapter(FitProblem)
ta.rebuild()

schema = {"$schema": "https://json-schema.org/draft-07/schema#", "$id": "refl1d-draft-01"}
# schema.update(get_model(FitProblem).schema())
schema.update(ta.json_schema(schema_generator=CustomsGenerateJsonSchema))


def remove_default_typename(schema):
    properties = schema.get("properties", {})
    properties.get("type", {}).pop("default", None)
    subschemas = schema.get("definitions", {}).values()
    for subschema in subschemas:
        remove_default_typename(subschema)


def remove_proptitles(schema):
    properties = schema.get("properties", {})
    for prop in properties.values():
        prop.pop("title", None)

    subschemas = schema.get("definitions", {}).values()
    for subschema in subschemas:
        remove_proptitles(subschema)


def convert_inf(obj):
    if isinstance(obj, (list, tuple)):
        return type(obj)(convert_inf(v) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((convert_inf(k), convert_inf(v)) for k, v in obj.items())
    elif isinstance(obj, float):
        return str(obj) if math.isinf(obj) else obj
    elif isinstance(obj, int) or isinstance(obj, str) or obj is None:
        return obj
    else:
        raise ValueError("obj %s is not recognized" % str(obj))


remove_default_typename(schema)
remove_proptitles(schema)
schema = convert_inf(schema)

os.makedirs("schema", exist_ok=True)
open("schema/refl1d.schema.json", "w").write(json.dumps(schema, allow_nan=False, indent=2))
