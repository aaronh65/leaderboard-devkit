import numpy as np
from env_utils import *

# signed angle difference w/target angle t2 and reference angle t1
def sgn_angle_diff(t1, t2):
    diff = t2 - t1
    diff = (diff + 180) % 360 - 180
    return diff
