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

# TORCH RELEVANT
import torch

@torch.no_grad()
# assumes n,c,h,w
def spatial_norm(tensor):
    n,c,h,w = tensor.shape
    flat = tensor.view((n,c,h*w))
    norm_max, _ = torch.max(flat, dim=-1, keepdim=True)
    norm_min, _ = torch.min(flat, dim=-1, keepdim=True)
    flat = (flat - norm_min) / (norm_max - norm_min)
    out = flat.view_as(tensor)
    return out # n,c,h,w

# given NxCxHxW and NxCx2 coordinates, retrieve NxCx1 values
def spatial_select(tensor, coord):
    n,c,h,w = tensor.shape
    coord = coord.long()
    x,y = coord[...,0], coord[...,1] #N,4
    flat_idx = torch.unsqueeze(y*w+x, -1)
    res = torch.gather(tensor.view((n,c,h*w)), -1, flat_idx)
    return res #N,C,1

def encode_str(string):
    return [ord(c) for c in string]

def decode_str(string):
    return ''.join(chr(c) for c in string)



