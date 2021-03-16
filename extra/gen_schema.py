import os
os.environ['BUMPS_USE_PYDANTIC'] = "True"

from pydantic.schema import get_model, schema as make_schema
from bumps.parameter import UnaryExpression, Expression
from refl1d.names import *
from refl1d.model import Repeat, Stack

base_model = get_model(FitProblem)

# resolve circular dependencies and self-references
# TODO: this will be unnecessary in python 3.7+ with
#     'from __future__ import annotations'
# and in python 4.0+ presumably that can be removed as well.
to_resolve = [
    UnaryExpression, Expression,
    Repeat, Stack
]
for module in to_resolve:
    get_model(module).update_forward_refs()

schema = make_schema([FitProblem])
