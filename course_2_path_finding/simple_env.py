import shapely
import numpy as np
import matplotlib.pyplot as plt


class SimpleEnv():
    def __init__(self, bounds):
        self.objects = None
        self.bounds = bounds

    def add_polygon(self, vertices_coords, holes_coords=None):
        pol = shapely.Polygon(vertices_coords, holes_coords)
        self.objects = shapely.unary_union([self.objects, pol])

    def add_disc(self, centers_coords: np.array, radius: float, N_vertices=15):
        assert centers_coords.shape == (2,)

        theta = 2*np.pi/N_vertices
        ori_point_coords = np.array([radius, 0])
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        points_coords = [(centers_coords+np.linalg.matrix_power(
            rotation_matrix, n) @ ori_point_coords) for n in range(N_vertices)]
        disc = shapely.Polygon(points_coords)
        self.objects = shapely.unary_union([self.objects, disc])

    def points_are_in_an_object(self, points_coords: np.ndarray):
        '''
        Args:
         - points_coords: np.ndarray of shape (n_points,2)
            coordinates of the points to probe
        Returns:
         - True if the points defined in points_coords intersect with any object of the environment
        '''
        points = shapely.MultiPoint(points_coords)
        return not (points.disjoint(self.objects))

    def segments_are_in_an_object(self, segments_coords):
        '''
        Args:
         - segments_coords: np.ndarray of shape (n_segments,2,2)
            coordinates of the segments to probe (segments_coords[0] = [first_point_coords,second_point_coords])
        Returns:
         - True if the segments defined in segments_coords intersect with any object of the environment
        '''
        segments = shapely.MultiLineString(segments_coords)
        return not (segments.disjoint(self.objects))

    def plot_objects(self, title="", alpha=0.5, fc='b', graph=None):
        import networkx as nx

        if self.objects is None:
            return None
        else:
            fig, axs = plt.subplots()
            axs.set_aspect('equal', 'datalim')

            for geom in self.objects.geoms:
                xs, ys = geom.exterior.xy
                axs.fill(xs, ys, alpha=alpha, fc=fc, ec='none')

            if graph is not None:
                node_data_0 = next(iter(graph.nodes.data()))[-1]
                edge_data_0 = next(iter(graph.nodes.data()))[-1]

                node_color, edge_color = None, None

                if "color" in node_data_0.keys():
                    node_color = list(nx.get_node_attributes(
                        graph, "color").values())

                if "color" in edge_data_0.keys():
                    edge_color = list(nx.get_edge_attributes(
                        graph, "color").values())

                nx.draw_networkx(graph, pos=nx.get_node_attributes(graph, "coords"), node_size=5,
                                 with_labels=False, node_color=node_color, edge_color=edge_color, arrows=False)

            plt.xlim(self.bounds[0], self.bounds[2])
            plt.ylim(self.bounds[1], self.bounds[-1])
            plt.title(title)
            plt.show()
