import os

def mkdir_if_not_exists(path, verbose=False):
    if not os.path.exists(path):
        if verbose:
            print(f'Creating a directory at {path}')
        os.makedirs(path)


