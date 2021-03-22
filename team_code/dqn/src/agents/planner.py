import os
from collections import deque

import numpy as np


DEBUG = int(os.environ.get('HAS_DISPLAY', 0))


class Plotter(object):
    def __init__(self, size):
        self.size = size
        self.clear()
        self.title = str(self.size)

    def clear(self):
        from PIL import Image, ImageDraw

        self.img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.img)

    def dot(self, pos, node, color=(255, 255, 255), r=2):
        x, y = 5.5 * (pos - node)
        x += self.size / 2
        y += self.size / 2

        self.draw.ellipse((x-r, y-r, x+r, y+r), color)

    def show(self):
        if not DEBUG:
            return

        #import cv2
        #print(f'display title {self.title}')
        #cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        #cv2.waitKey(1)


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=256):
        self.route = deque() # in meters
        self.min_distance = min_distance # default 7.5
        self.max_distance = max_distance # default 25.0

        #self.mean = np.array([49.0, 8.0])
        #self.scale = np.array([111324.60662786, 73032.1570362])
        self.mean = np.array([0.0, 0.0])
        self.scale = np.array([111324.60662786, 111324.60662786])

        self.debug = Plotter(debug_size)

    def set_route(self, global_plan, gps=False):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps: # from lat/lon to meters
                pos = np.array([pos['lat'], pos['lon']])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean
            #print(f'cmd = {cmd}\npos = {pos}')
            self.route.append((pos, cmd))

    def run_step(self, gps):
        self.debug.clear()

        if len(self.route) == 1:
            return self.route[0]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        # search for next waypoint within max range of 25.0m
        for i in range(1, len(self.route)):

            # break if we've searched far enough along route
            if cumulative_distance > self.max_distance:
                break

            # measures along route distance
            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            # measure distance from waypoint i to current location
            distance = np.linalg.norm(self.route[i][0] - gps)

            # if waypoint is within min distance of current location
            # and it's the farthest one from current location
            # then select
            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

            ## red if we're far, green if command is ???
            #r = 255 * int(distance > self.min_distance)
            ##g = 255 * int(self.route[i][1].value == 4)
            #g = 0
            #b = 255
            #self.debug.dot(gps, self.route[i][0], (r, g, b))

        # discard the 
        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()
        
        self.debug.dot(gps, self.route[0][0], (0, 255, 0))
        self.debug.dot(gps, self.route[1][0], (255, 0, 0))
        for i in range(2, min(5, len(self.route))):
            self.debug.dot(gps, self.route[i][0], (139,0,139))
        self.debug.dot(gps, gps, (255, 255, 255))
        self.debug.show()

        return self.route
        #return self.route[0], self.route[1]
