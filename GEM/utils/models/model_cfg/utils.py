import importlib.util
import inspect
import pdb

def read_model_cfg(cfg_path):
    module_name = cfg_path.split('/')[-1].split('.')[0]
    spec = importlib.util.spec_from_file_location(module_name, cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = {}
    for name in dir(module):
        obj = getattr(module, name)
        if not name.startswith('__'):
            if inspect.isclass(obj) or inspect.isfunction(obj) or not inspect.ismodule(obj):
                result[name] = obj
    return result