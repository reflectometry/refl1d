import importlib.util


def clone_module(original_module: str):
    spec = importlib.util.find_spec(original_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
