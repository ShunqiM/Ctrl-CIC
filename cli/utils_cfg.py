from typing import Any, Callable, Dict, List, Tuple, Union

def load_config(fpath: str, recursive_flag='recursive_load_config', base_config_name="_default.yaml"):
    """Load a config file from a path. 
    (set recursive_load_config=True in config.yaml, it will auto-load the _default.yaml file in ALL parent directories (i.e, shared config files)))
    """
    
    ### let yaml recognize ${ENV_VAR} syntax (for example, ${CKPT} and ${PREP})
    import yaml, re, os
    path_matcher = re.compile(r'\$\{([^}^{]+)\}')
    def path_constructor(loader, node):
        ''' Extract the matched value, expand env variable, and replace the match '''
        value = node.value
        match = path_matcher.match(value)
        env_var = match.group()[2:-1]
        env_val = os.environ.get(env_var)
        print(f'recognized env variable (${env_var}) with value ("{env_val}") in config file ({fpath})')
        return env_val + value[match.end():]
    yaml.add_implicit_resolver('!path', path_matcher)
    yaml.add_constructor('!path', path_constructor)
    def pjoin(loader, node):
        seq = loader.construct_sequence(node)
        return '/'.join([str(i) for i in seq])
    yaml.add_constructor('!pjoin', pjoin)

    if not os.path.exists(fpath): raise FileNotFoundError(fpath)

    fpaths = [fpath]

    # quick peek at the fpath to see if it has a "recursive_load" flag
    recursive_load = yaml.load(open(fpath), Loader=yaml.FullLoader).pop(recursive_flag, False)
    if recursive_load:
        while os.path.dirname(fpath) != fpath:
            fpath = os.path.dirname(fpath)
            fpaths.append(os.path.join(fpath, base_config_name))

    def update_config_dict(config: Dict, other: Dict) -> Dict:
        for key, value in other.items():
            if isinstance(value, dict):
                if key not in config or not isinstance(config[key], Dict):
                    config[key] = {}
                config[key] = update_config_dict(config[key], value)
            else:
                config[key] = value
        return config

    config = {}
    for fpath in reversed(fpaths):
        if not os.path.exists(fpath): continue

        with open(fpath) as f:
            curr_config = yaml.load(f, Loader=yaml.FullLoader)
        #print(fpath, curr_config, '\n')
        config = update_config_dict(config, curr_config)

    config.pop(recursive_flag, None)     
    return config

def override_config(config: Dict, override_params: List[Tuple[str, Any]]):
    """Overwrite specific params passed as command line args."""
    for arg, value in override_params:
        current_level = config
        arg_parts = arg.split(".")
        for j, part in enumerate(arg_parts):
            if j == len(arg_parts) - 1:
                if part in current_level: 
                    current_level[part] = value
                    break
            if part not in current_level:
                if j == len(arg_parts) - 1:
                    current_level[part] = value
                else:
                    current_level[part] = {}
            current_level = current_level[part]


            