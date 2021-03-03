import refl1d.names
import bumps.bounds
import bumps.fitproblem

from dataclasses import is_dataclass, fields

def get_dataclass_defs(sources = (refl1d.names, bumps.bounds, bumps.fitproblem)):
    class_defs = {}
    for source in sources:
        names = dir(source)
        dataclasses = dict([(n, getattr(source, n)) for n in names if is_dataclass(getattr(source, n))])
        class_defs.update(dataclasses)
    return class_defs

CLASS_DEFS = get_dataclass_defs()

class Deserializer(object):
    def __init__(self, class_defs=CLASS_DEFS):
        self.refs = {}
        self.deferred = {}
        self.class_defs = class_defs

    def rehydrate(self, obj):
        if isinstance(obj, dict):
            obj = obj.copy()
            t = obj.pop('type', None)
            for key,value in obj.items():
                obj[key] = self.rehydrate(value)
                #print(key)
            if t in self.class_defs:
                hydrated = self.instantiate(t, obj)
                return hydrated
            else:
                return obj
        elif isinstance(obj, list):
            return [self.rehydrate(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.rehydrate(v) for v in obj)
        else:
            return obj

    def instantiate(self, typename, serialized):
        s = serialized.copy()
        id = s.get("id", None) # will id be a required schema element?
        #print('rehydrating: ', typename)
        if id is None or not id in self.refs:
            class_factory = self.class_defs.get(typename)
            if hasattr(class_factory, 'from_dict'):
                class_factory = class_factory.from_dict
            hydrated = class_factory(**s)
            if id is not None:
                self.refs[id] = hydrated
        else:
            hydrated = self.refs[id]
        return hydrated

def from_dict(serialized):
    oasis = Deserializer()
    return oasis.rehydrate(serialized)

import copy
import numpy as np

def to_dict(obj):
    if is_dataclass(obj):
        result = [('type', obj.__class__.__name__)]
        for f in fields(obj):
            value = to_dict(getattr(obj, f.name))
            result.append((f.name, value))
        return dict(result)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_dict(v) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((to_dict(k), to_dict(v))
                          for k, v in obj.items())
    elif isinstance(obj, np.ndarray):
        #return dict(type="numpy.ndarray", dtype=obj.dtype.name, values=obj.tolist())
        return obj.tolist()
    else:
        return copy.deepcopy(obj)
