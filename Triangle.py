import numpy as np
from gymnasium import Env, spaces
import matplotlib.pyplot as plt
from typing import Callable, Optional
from gymnasium.core import ObsType, ActType

class Triangle:

    def __init__(self, points, max_side_length, box_min, box_max):
        """
        constract a triangle inside a bounding box.

        Parameters:
        points: is the apex point of the triangle or
        points (list of tuples): [(x1, y1), (x2, y2), (x3, y3)]
        max_side_length: is the maximum side length of the equilateral triangle
        box_min (tuple): (xmin, ymin)
        box_max (tuple): (xmax, ymax)
        """

        self.max_side_length = max_side_length
        self.box_min = box_min
        self.box_max = box_max

        if len(points) < 3:
            self.apex_point = points
            self.triangle_vertices = self.get_triangle(self.apex_point,
                                                       max_side_length)
        else:
            self.apex_point = points[0]
            self.triangle_vertices = points

    def rotate(self, angle_rad):
        """
        Rotate a point counterclockwise by a given angle around a given apex point.
        The angle should be given in radians.
        Returns:
        Rotated triangle
        """

        rtriangle_vertices = np.zeros((0, 2))
        for point in self.triangle_vertices.copy():
            # Rotation matrix
            rotation_matrix = np.array(
                [[np.cos(angle_rad), -np.sin(angle_rad)],
                 [np.sin(angle_rad), np.cos(angle_rad)]])

            # Translate point to origin
            translated_point = point - self.apex_point

            # Apply rotation
            rotated_point = np.dot(rotation_matrix, translated_point)

            # Translate point back
            tpoint = rotated_point + self.apex_point
            rtriangle_vertices = np.vstack((rtriangle_vertices, tpoint))

        return Triangle(rtriangle_vertices, self.max_side_length, self.box_min,
                        self.box_max)

    def plot(self, color='b'):
        """
        Plot a triangle given the vertices.
        """
        plt.scatter(self.apex_point[0], self.apex_point[1], color=color)
        vertices = np.vstack(
            (self.triangle_vertices, self.triangle_vertices[0]))
        plt.plot(vertices[:, 0], vertices[:, 1], color=color)

    def is_valid(self):
        """Check if the triangle is entirely inside a bounding box. """
        for vertex in self.triangle_vertices:
            x, y = vertex
            if not (self.box_min[0] <= x <= self.box_max[0]
                    and self.box_min[1] <= y <= self.box_max[1]):
                return False
        return True

    def get_angle(self):
        A, B, C = self.triangle_vertices
        midpoint = (B + C) / 2
        relDiff = (A - midpoint)
        return np.arctan2(relDiff[1], relDiff[0])

    def get_triangle(self, initial_point, max_side_length):
        """
        Compute the equilateral triangle given an apex point and max side length
        """
        # Calculate other two vertices
        third_point = initial_point - np.array(
            [max_side_length / 2,
             np.sqrt(3) / 2 * max_side_length])
        second_point = third_point + np.array([max_side_length, 0])

        triangle_vertices = np.vstack(
            (initial_point, second_point, third_point))
        return triangle_vertices


class HumanPolicy:

    def __init__(self, x, y, vx, vy, width, height):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.width = width
        self.height = height
        self.yaw = None
        self.offset = 0.2

    def update(self):
        self.x += self.vx
        self.y += self.vy

        if self.x > self.width - self.offset  or self.x < self.offset :
            self.vx *= -1

        if self.y > self.height - self.offset  or self.y < self.offset :
            self.vy *= -1



class TriangleEnv(Env):
    def __init__(self, max_side_length, box_min, box_max):
        self.max_side_length = max_side_length
        self.box_min = box_min
        self.box_max = box_max
        self.width = box_max[0] - box_min[0]
        self.height = box_max[1] - box_min[1]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-min(box_min), high=max(box_max), shape=(10,), dtype=np.float32)  # TODO: increase the state

    def map_to_0_2pi(self, value):
        if value < -1 or value > 1:
            raise ValueError("Input value must be in the range [-1, 1]")
        return (value + 1) * np.pi

    def step(self, action):

        self.point.update()
        apex_point = np.array([self.point.x, self.point.y])
        self.tri = Triangle(apex_point, self.max_side_length, self.box_min, self.box_max)
        self.tri = self.tri.rotate(self.map_to_0_2pi(action[0]))


        s_t = self.tri.triangle_vertices.tolist()
        s_t.append(self.box_min)
        s_t.append(self.box_max)
        s_t = np.array(s_t, dtype=np.float32)

        # compute reward
        r_t = 1000.0 * np.exp(-abs(self.last_triangle.get_angle() - self.tri.get_angle()))
        done = not self.tri.is_valid()
        if done: r_t = -1000.0
        done = self.step_count > 10000

        self.step_count += 1
        self.last_triangle = self.tri

        return s_t.flatten(), r_t, done, True, {}


    def reset(self,
              *,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None):
        self.point = HumanPolicy(x=2, y=3, vx=0.03, vy=0.06, width=self.width, height=self.height)
        self.last_triangle = Triangle(np.array([2, 3]), self.max_side_length, self.box_min, self.box_max)
        self.step_count = 0
        s_t = self.last_triangle.triangle_vertices.tolist()
        s_t.append(self.box_min)
        s_t.append(self.box_max)
        s_t = np.array(s_t, dtype=np.float32)
        return s_t.flatten(), {}

    def render(self, mode='human'):

        plt.cla()
        self.tri.plot(color='b') if self.tri.is_valid() else self.tri.plot(color='r')
        plt.title('heading angle {0:.3f} deg'.format(
            np.rad2deg(self.tri.get_angle())))
        plt.xlim(self.box_min[0], self.box_max[0])
        plt.ylim(self.box_min[1], self.box_max[1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.pause(0.001)

