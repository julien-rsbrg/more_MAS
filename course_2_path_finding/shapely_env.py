import shapely
import numpy as np


class ShapelyEnv():
    def __init__(self):
        self.objects = None

    def add_polygon(self, vertices_coords):
        pol = shapely.Polygon(vertices_coords)
        self.objects = shapely.GeometryCollection([self.objects, pol])

    def add_disc(self, centers_coords: np.array, radius: float, N_vertices=15):
        assert centers_coords.shape == (2,)

        theta = 2*np.pi/N_vertices
        ori_point_coords = np.array([radius, 0])
        print("ori_point_coords", ori_point_coords)

        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        points_coords = [(centers_coords+np.linalg.matrix_power(
            rotation_matrix, n) @ ori_point_coords) for n in range(N_vertices)]
        print("points_coords", points_coords)
        disc = shapely.Polygon(points_coords)
        self.objects = shapely.GeometryCollection([self.objects, disc])

    def is_in_an_object(self, X, Y):
        if isinstance(X, list) or isinstance(X, np.ndarray):
            coords = [[X[i], Y[i]] for i in range(len(X))]
            points = shapely.MultiPoint(coords)
            return not (points.disjoint(self.objects))
        else:
            point = shapely.Point((X, Y))
            return not (point.disjoint(self.objects))
