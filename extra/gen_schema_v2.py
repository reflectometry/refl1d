import json

from bumps.parameter import *
from bumps.util import NumpyArray
from pydantic_core import core_schema
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic import TypeAdapter

from refl1d.names import *
from refl1d.model import *
from refl1d.fitproblem import FitProblem

NDArray = NumpyArray


class BumpsGenerateJsonSchema(GenerateJsonSchema):
    def dataclass_schema(self, schema: core_schema.DataclassSchema) -> JsonSchemaValue:
        cls = schema["cls"]
        fqn = f"{cls.__module__}.{cls.__name__}"
        print("fqn: ", fqn)
        # filter attrs with leading underscore name (same behavior as in v1)
        schema["schema"]["fields"] = [f for f in schema["schema"]["fields"] if not f["name"].startswith("_")]
        json_schema = super().dataclass_schema(schema)
        properties = json_schema.get("properties", {})
        for prop, value in properties.items():
            value.pop("title", None)
        properties.setdefault("type", {"enum": [fqn]})
        return json_schema


TA = TypeAdapter(FitProblem)
schema = TA.json_schema(schema_generator=BumpsGenerateJsonSchema)

print(json.dumps(schema, indent=2))
