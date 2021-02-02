import carla
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


''' object transformations '''
# carla 3d vector to np array
def cvector_to_array(carla_vector):
    v = carla_vector
    return np.array([v.x, v.y, v.z])

# carla Transform to 6-d vector
def transform_to_vector(transform):
    loc = transform.location
    rot = transform.rotation
    return np.array([loc.x, loc.y, loc.z, rot.pitch, rot.yaw, rot.roll])

# 6d vector to carla Transform
def vector_to_transform(vector):
    x,y,z,pitch,yaw,roll = vector
    loc = carla.Location(x,y,z)
    rot = carla.Rotation(pitch,yaw,roll)
    return carla.Transform(loc, rot)

# carla Waypoint to 6d vector
def waypoint_to_vector(waypoint):
    transform = waypoint.transform
    return transform_to_vector(transform)

# 6d vector to carla Waypoint
def vector_to_waypoint(vector):
    transform = vector_to_transform(vector)
    return carla.Waypoint(transform)


''' given a ref and target transform, return a transform that goes from
the reference frame to the target frame '''
def chain_transform(ref_transform, tgt_transform):
    ref2world = np.array(ref_transform.get_matrix())
    world2tgt = np.array(tgt_transform.get_inverse_matrix())
    ref2tgt = np.matmul(world2tgt, ref2world)
    return ref2tgt

# signed angle difference w/target angle t2 and reference angle t1
def sgn_angle_diff(t1, t2):
    diff = t2 - t1
    diff = (diff + 180) % 360 - 180
    return diff

''' adding transforms together '''
def add_location(location, dx=0, dy=0, dz=0):
    return carla.Location(
            location.x + dx,
            location.y + dy,
            location.z + dz)

def add_rotation(rotation, dp=0, dy=0, dr=0):
    return carla.Rotation(
            rotation.pitch + dp,
            rotation.yaw + dy,
            rotation.roll + dr)

def add_transform(transform, dx=0, dy=0, dz=0, dp=0, dyaw=0, dr=0):
    location = add_location(transform.location,dx,dy,dz)
    rotation = add_rotation(transform.rotation,dp,dyaw,dr)
    return carla.Transform(location, rotation)


''' visualizations '''
def draw_transforms(world, transforms, color=(255,0,0), z=0.5, life_time=0.06, size=0.3):
    r,g,b = color
    ccolor = carla.Color(r,g,b)
    for tf in transforms:
        begin = tf.location + carla.Location(z=z)
        angle = math.radians(tf.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=size, life_time=life_time, color=ccolor, thickness=size)

def draw_waypoints(world, waypoints, color=(255,0,0), z=0.5, life_time=0.06, size=0.3):
    transforms = [wp.transform for wp in waypoints]
    draw_transforms(world, transforms, color, z, life_time=life_time, size=size)

def draw_arrow(world, loc1, loc2, color=(255,0,0), z=0.5, life_time=0.06, size=0.3):
    r,g,b = color
    ccolor = carla.Color(r,g,b)
    world.debug.draw_arrow(loc1, loc2, arrow_size=size, life_time=life_time, color=ccolor, thickness=size)

