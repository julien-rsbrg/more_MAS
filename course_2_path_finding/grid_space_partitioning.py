import numpy as np
import networkx as nx
import copy


def generate_grid_graph(env, grid_bounds, nelemx, nelemy):
    xmin, ymin, xmax, ymax = grid_bounds
    X = np.linspace(xmin, xmax, nelemx)
    Y = np.linspace(ymin, ymax, nelemy)
    X, Y = np.meshgrid(X, Y)

    coords = {}
    colors = {}
    for j in range(nelemx):
        for i in range(nelemy):
            coords[j, i] = np.array([X[i, j], Y[i, j]])
            # colors[j,i] = int(X[i,j] >= 0.5)
            colors[j, i] = "blue"

    G = nx.grid_2d_graph(nelemx, nelemy)
    G = G.to_undirected()
    nx.set_node_attributes(G, coords, "coords")
    nx.set_node_attributes(G, colors, "colors")

    # add the distances between the nodes. Inf if it leads to an object in env
    measure_edge_distances(env, G)
    get_undirect_edge_attribute(G, "distances")

    return G


def check_available(env, graph):
    """
    Probe an environment
    """
    available = {}
    for node_id, node_data in graph.nodes.data():
        node_coords = node_data["coords"]

        available[node_id] = not (env.points_are_in_an_object(
            [node_coords]))
    nx.set_node_attributes(graph, available, "available")


def measure_edge_distances(env, graph):
    """
    Add "distances" between adjacent nodes to graph as an edge attribute
    """
    distances = {}
    node_coords = nx.get_node_attributes(graph, "coords")

    # probe the environment to know where an object is
    check_available(env, graph)

    node_availibilities = nx.get_node_attributes(graph, "available")
    for edge_id in graph.edges:
        node_0, node_1 = edge_id
        if not (node_availibilities[node_1]):
            distances[edge_id] = np.inf
        else:
            node_coord_0 = node_coords[node_0]
            node_coord_1 = node_coords[node_1]
            dist = np.linalg.norm(node_coord_1-node_coord_0)
            distances[edge_id] = dist

    nx.set_edge_attributes(graph, distances, "distances")
    # print(graph.edges.data())
    # return graph


def get_undirect_edge_attribute(graph, attribute_name):
    """
    Ensure the attribute is undirected
    """
    attribute = nx.get_edge_attributes(graph, attribute_name)

    both_direction_info = False
    new_attribute = copy.deepcopy(attribute)
    for edge_id, edge_data in attribute.items():
        if edge_id[::-1] in attribute:
            both_direction_info = True
        new_attribute[edge_id[::-1]] = edge_data

    if both_direction_info:
        print("Already infomation in both direction")
    nx.set_edge_attributes(graph, new_attribute, attribute_name)
    return new_attribute


def color_path_and_graph(env, graph, path):
    """
    Change the color in graph to show the path to take
    """
    color = {}
    for node_id, node_data in graph.nodes.data():
        node_coords = node_data["coords"]
        color[node_id] = "red" if env.points_are_in_an_object(
            [node_coords]) else "blue"
        if node_id in path:
            color[node_id] = "green"
    nx.set_node_attributes(graph, color, "color")
