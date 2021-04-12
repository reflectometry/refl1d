import os
import math
import json
os.environ['BUMPS_USE_PYDANTIC'] = "True"

from pydantic.schema import get_model, schema as make_schema
from bumps.parameter import Expression, Parameter #, UnaryExpression
from refl1d.fitproblem import FitProblem
from refl1d.model import Repeat, Stack

base_model = get_model(FitProblem)

# resolve circular dependencies and self-references
# TODO: this will be unnecessary in python 3.7+ with
#     'from __future__ import annotations'
# and in python 4.0+ presumably that can be removed as well.
to_resolve = [
    Expression, Parameter, #UnaryExpression
    Repeat, Stack
]
for module in to_resolve:
    get_model(module).update_forward_refs()

schema = {
    "$schema": "https://json-schema.org/draft-07/schema#",
    "$id": "refl1d-draft-01"
}
schema.update(get_model(FitProblem).schema())

def remove_default_typename(schema):
    properties = schema.get('properties', {})
    properties.get('type', {}).pop('default', None)
    subschemas = schema.get('definitions', {}).values()
    for subschema in subschemas:
        remove_default_typename(subschema)

def remove_proptitles(schema):
    properties = schema.get('properties', {})
    for prop in properties.values():
        prop.pop('title', None)
    
    subschemas = schema.get('definitions', {}).values()
    for subschema in subschemas:
        remove_proptitles(subschema)

def convert_inf(obj):
    if isinstance(obj, (list, tuple)):
        return type(obj)(convert_inf(v) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((convert_inf(k), convert_inf(v))
                          for k, v in obj.items())
    elif isinstance(obj, float):
        return str(obj) if math.isinf(obj) else obj
    elif isinstance(obj, int) or isinstance(obj, str) or obj is None:
        return obj
    else:
        raise ValueError("obj %s is not recognized" % str(obj))


remove_default_typename(schema)
remove_proptitles(schema)
schema = convert_inf(schema)

open('schema/refl1d.schema.json', 'w').write(json.dumps(schema, allow_nan=False, indent=2))

