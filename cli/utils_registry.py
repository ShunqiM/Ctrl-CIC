from typing import Any, Callable, Dict, List, Tuple, Union
from pydprint import dprint as dp
from utils.utils_color import colorize

def import_class_by_full_name(module_class_name: str) -> Any:
    import importlib

    """Import and return a class from its full string name."""
    module_parts = module_class_name.split(".")
    module_name = ".".join(module_parts[0:-1])
    class_name = module_parts[-1]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)

    return class_

class Registry:
    _registered_stuff = {} # callables (func or class)

    @classmethod
    def register(cls, stuff_name: str) -> Callable:
        # stuff could be class or function
        def inner(stuff_to_register: Any) -> Any:
            cls._registered_stuff[stuff_name] = stuff_to_register
            return stuff_to_register
        return inner
    
    @classmethod
    def getter(cls, name: str) -> Any:
        if name == 'PLACEHOLDER':
            print(f"A {colorize('callable', 'red')} is found to be {colorize('PLACEHOLDER', 'red')}. check config file. exiting...")
            exit()

        if name in cls._registered_stuff:
            return cls._registered_stuff[name]
        else:
            try:
                return import_class_by_full_name(name)
            except ValueError:
                raise ValueError(f"Could not find '{name}' in either registry or import path. Make sure it's properly imported (e.g., __init__.py)")
   
    @classmethod
    def convert_cfg_node(cls, root_node: Dict) -> None:
        if not isinstance(root_node, dict): 
            return 
        
        if "callable" in root_node: # only a single model/dataset/optimizer/collator/etc.
            root_node["callable"] = cls.getter(root_node["callable"])

        for name, sub_node in root_node.items():
            cls.convert_cfg_node(sub_node)


    @classmethod
    def build_instance_from_cfg_node(cls, cfg_node: Dict, **other_params) -> object:
        assert "callable" in cfg_node, f"cfg_node must have a 'callable' key to build instance, got {cfg_node}"
        _callable = cfg_node["callable"]
        _params = cfg_node.get("params", {})
        _params = _params or {}
        _params.update(other_params)
        return _callable(**_params)
    
