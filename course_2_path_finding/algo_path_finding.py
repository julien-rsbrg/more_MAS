# standard librairies
import networkx as nx
import numpy as np

# from this library
import course_2_path_finding.grid_space_partitioning as grd_prt


def path_finding_djisktra(graph: nx.Graph, start_node, end_node):
    """
    Apply djikstra algorithm to path finding in graph
    """
    n_nodes_graph = graph.number_of_nodes()
    # ensure the
    graph_distances = grd_prt.get_undirect_edge_attribute(
        graph, "distances")

    # initialize distances to start_node
    path_distances = {}
    path_distances[start_node] = 0
    for node in graph.nodes:
        if node != start_node:
            path_distances[node] = np.inf

    predecessor = {}
    # explore
    found_end_node = start_node == end_node
    nodes_taken = set()
    while len(predecessor) < n_nodes_graph and not (found_end_node):
        # find the closest node
        dist_closest = np.inf
        closest_node = None
        for node, dist in path_distances.items():
            if dist < dist_closest and node not in nodes_taken:
                dist_closest, closest_node = dist, node

        # as the weights are positive
        if closest_node == end_node:
            found_end_node = True

        # add it to the taken nodes
        nodes_taken.add(closest_node)
        # update the neighbour nodes path distances
        for neighbor_node in graph.neighbors(closest_node):
            if path_distances[neighbor_node] > path_distances[closest_node] + graph_distances[(closest_node, neighbor_node)]:
                path_distances[neighbor_node] = path_distances[closest_node] + \
                    graph_distances[(closest_node, neighbor_node)]
                predecessor[neighbor_node] = closest_node

    # reconstruct the path to take
    prev_node = end_node
    path = [prev_node]
    while prev_node != start_node:
        prev_node = predecessor[prev_node]
        path.append(prev_node)

    path = path[::-1]

    return path
