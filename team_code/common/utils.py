import os
from types import SimpleNamespace

def mkdir_if_not_exists(path, verbose=False):
    if not os.path.exists(path):
        if verbose:
            print(f'Creating a directory at {path}')
        os.makedirs(path)

# transforms a dict to a namespace
class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

# recursively transforms a dict to a namespace
def dict_to_sns(d):
    return SimpleNamespace(**d)

def parse_config(maybe_config):
    config_type = type(maybe_config)
    if config_type == str:
        with open(maybe_config, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config = dict_to_sns(config)
    elif config_type == dict:
        config = dict_to_sns(maybe_config)
    elif config_type == SimpleNamespace:
        config = maybe_config
    else:
        raise Exception('config type not understood')
    return config

def port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

