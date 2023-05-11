import os
import yaml
import logging

def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_specific = yaml.load(f, Loader=yaml.SafeLoader)

    # Check if we should inherit from a config
    inherit_from = cfg_specific.get('inherit_from')

    # If yes, load this config first as default
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    # If no, use the default_path
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        cfg = dict()

    # update cfg using cfg_specific
    update_recursive(cfg, cfg_specific)

    return cfg

def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

def setup_logging(out_dir):
    # If modules imported before this has already configured basicConfig, we cannot set it here again.
    # Therefore we reload it. 
    # Since python 3.8, can skip the reload and set force=True for basicConfig.
    # https://stackoverflow.com/questions/20240464/python-logging-file-is-not-working-when-using-logging-basicconfig
    from importlib import reload # python 2.x don't need to import reload, use it directly
    reload(logging)

    # Set up the logging format and output path
    level    = logging.INFO
    format   = '%(asctime)s %(message)s'
    datefmt = '%m-%d %H:%M:%S'
    logfile = os.path.join(out_dir, 'msgs.log')
    handlers = [logging.FileHandler(logfile), logging.StreamHandler()]
    
    logging.basicConfig(level = level, format = format, datefmt=datefmt, handlers = handlers)
    logging.info('Hey, logging is written to {}!'.format(logfile))
    return

# def setup_logging(out_dir):
#     # Set up the logging format and output path
#     level    = logging.INFO
#     format   = '%(asctime)s %(message)s'
#     datefmt = '%m-%d %H:%M:%S'

#     formatter = logging.Formatter(format, datefmt)
#     handler_file = logging.FileHandler(os.path.join(out_dir, 'msgs.log'))
#     handler_file.setFormatter(formatter)
#     handler_stream = logging.StreamHandler()
#     handler_stream.setFormatter(formatter)

#     logger = logging.getLogger()
#     logger.addHandler(handler_file)
#     logger.addHandler(handler_stream)
#     logger.setLevel(level)

#     logger.info('Hey, logging is written to {}!'.format(os.path.join(out_dir, 'msgs.log')))
#     return