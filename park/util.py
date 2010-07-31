import sys

def import_symbol(path):
    """
    Recover symbol from path.
    """
    parts = path.split('.')
    module_name = ".".join(parts[:-1])
    symbol_name = parts[-1]
    __import__(module_name)
    module = sys.modules[module_name]
    symbol = getattr(module,symbol_name)
    #if not hasattr(symbol, 'park_export'):
    #    raise TypeError("Symbol %s not marked @park.export"%path)
    return symbol

