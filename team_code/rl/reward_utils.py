import numpy as np
from env_utils import *

# signed angle difference w/target angle t2 and reference angle t1
def sgn_angle_diff(t1, t2):
    diff = t2 - t1
    diff = (diff + 180) % 360 - 180
    return diff

# index of closest transform in candidates (vector form)
def closest_aligned_transform(hero_transform, transforms, fvectors, world=None):

    # distance criteria
    hero_transform_vec = transform_to_vector(hero_transform)
    hero2pt = transforms[:,:3] - hero_transform_vec[:3]
    dist2pt = np.linalg.norm(hero2pt, axis=1)
    indices = np.argsort(dist2pt)

    # alignment criteria
    hero_fvec = hero_transform.get_forward_vector()
    hero_fvec = np.array([cvector_to_array(hero_fvec)]).T # 3x1

    alignment = np.matmul(hero2pt, hero_fvec).flatten()
    aligned = alignment > 0
    aligned = aligned[indices] # reorder by distance criteria
    indices = indices[aligned] # slice out valid indices

    targets = indices[:1]
    return targets
    
