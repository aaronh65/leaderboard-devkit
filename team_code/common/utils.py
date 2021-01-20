import os

def mkdir_if_not_exists(path, verbose=False):
    if not os.path.exists(path):
        if verbose:
            print(f'Creating a directory at {path}')
        os.makedirs(path)

# transforms a dict to a namespace
class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
