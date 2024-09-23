import random
import numpy as np
import torch
import json
from typing import Any, Callable, Dict, List, Tuple, Union
import src # registry
from cli.utils_registry import Registry
from transformers.utils import logging
from utils_cfg import load_config, override_config
from utils_parser import parse_args, parse_undefined_args
import pathlib
import os
import inspect
from comm_ddp import comm
from pydprint import dprint as dp

from utils.utils_color import colorize
DEFAULT_OUTPUT_DIR = 'tmp'

get_output_dir_from_pipline_config = lambda x: x['params']['output_dir']

def experiment_initialization():

    ### experiment mode: train or eval
    script_fname = get_parent_script_name()
    assert script_fname in ['train.py', 'eval.py'], f'unrecognized command-line-interface (CLI) script: {script_fname}'
    experiment_mode = script_fname.split('.')[0]
    assert experiment_mode in ['train', 'eval'], f'unrecognized experiment mode: {experiment_mode}'
    pipeline = 'trainer' if experiment_mode == 'train' else 'evaluator'

    ### parse command line args
    args, remaining_args = parse_args()
    override_params = parse_undefined_args(remaining_args)

    ### logger
    logger = config_logging()
    logger.info(f'Process rank: {comm.get_local_rank()}; distributed training: {comm.is_DDP_now()}')

    ### config 
    config = load_config(args.config) if args.config else {}
    override_config(config, override_params)
    # run_id = timestep + processid + random_word/specified word
    run_id = get_run_id(config['note'])
    if 'evaluator' in config and 'run_id' in config['evaluator']['params'] and config['evaluator']['params']['run_id'] is not None:
        run_id = config['evaluator']['params']['run_id']
    if 'trainer' in config and config['resume_name'] is not None:
        run_id = config['resume_name']
        # config[pipeline]['params']['output_dir'] = os.path.join(config[pipeline]['params']['output_dir'], run_id)
    config[pipeline]['params']['output_dir'] = os.path.join(config[pipeline]['params']['output_dir'], run_id)

    print(config[pipeline]['params']['output_dir'])
    update_unspecified_outdir(config, pipeline, run_id)
    if not args.wandb: config[pipeline]['params']['report_to'] = 'none'
    else:              config[pipeline]['params']['run_name'] = run_id


    print(config)
    logger.info(f'\nConfig from "{args.config}": {json.dumps(config, indent=2)}')
    logger.info(colorize(f"run_id: {run_id}", "red"))

    ### register callable
    Registry.convert_cfg_node(config) # Get the registered callables

    ### setup ckpt_dir
    create_ckpt_dir(config, pipeline, run_id) if comm.is_main_process() else 0
    
    ### saving temporary variable
    config['pipeline'] = pipeline
    return config, logger

def get_parent_script_name():
    frame = inspect.currentframe().f_back
    while frame:
        if frame.f_code.co_filename != __file__:
            return os.path.basename(frame.f_code.co_filename)
        frame = frame.f_back

    return None


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def config_logging():
    # Setup transformer logging
    log_level = logging.INFO if comm.is_main_process() else logging.WARNING
    logging.set_verbosity(log_level)
    logging.enable_default_handler()
    logging.enable_explicit_format()
    logging.enable_progress_bar()
    return logging.get_logger("transformers")

def get_run_id(note_string = None):

    def get_process_id():
        # process_id: pid or slurm_job_id
        process_id = os.environ.get('SLURM_JOB_ID', None)
        if process_id is None: process_id = f"pid{os.getpid()}"
        else: process_id = f"slurm{process_id}"
        return process_id

    def get_timestamp():
        from datetime import datetime
        now = datetime.now()
        return now.strftime("%y%m%d-%H%M%S")

    from random_word import RandomWords
    if note_string is None:
        return f'{get_timestamp()}-{get_process_id()}-{RandomWords().get_random_word()}'
    else:
        return f'{get_timestamp()}-{get_process_id()}-{note_string}-{RandomWords().get_random_word()}'

def update_unspecified_outdir(config, pipeline:str, run_id:str):
    
    output_dir = get_output_dir_from_pipline_config(config[pipeline])

    if output_dir == DEFAULT_OUTPUT_DIR:
        config[pipeline]['params']['output_dir'] = output_dir + f'/{run_id}' 

def create_ckpt_dir(config, pipeline:str, run_id:str):
    
    import yaml

    assert pipeline in ['trainer', 'evaluator'], pipeline
    if pipeline == 'trainer': config_name, log_name = 'train_config', 'train_log'
    else:                     config_name, log_name = 'eval_config', 'eval_log'

    # create dirs
    output_dir = get_output_dir_from_pipline_config(config[pipeline])

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    config_dir = output_dir / config_name
    config_dir.mkdir(exist_ok=True, parents=True)
    log_dir = output_dir / log_name
    log_dir.mkdir(exist_ok=True, parents=True)



    # save config
    with open(config_dir.joinpath(f"{config_name}.{run_id}.yaml"), "w") as fh:
        yaml.dump(config, fh)
    
    # link logger to txt
    def link_logger_to_txt():
        import logging
        file_handler = logging.FileHandler(log_dir.joinpath(f"{log_name}.{run_id}.txt")) 
        file_handler.setLevel(level=logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                                        datefmt="%m/%d/%Y %H:%M:%S", ))
        from transformers import logging
        logging.get_logger("transformers").addHandler(file_handler)
    link_logger_to_txt()
    
    return output_dir


def build_pipeline(config):
    assert 'pipeline' in config
    pipeline_name = config.pop('pipeline')
    assert pipeline_name in ['trainer', 'evaluator'], pipeline

    pipeline_config = config.pop(pipeline_name)
    pipeline = Registry.build_instance_from_cfg_node(pipeline_config)
    logging.get_logger("transformers").info(f'[rank:{comm.get_local_rank()}] creating pipeline ({pipeline_name}) as {pipeline.__class__.__name__}')
    
    pipeline.compile(**config)
    
    return pipeline