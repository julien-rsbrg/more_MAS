# standard libraries
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict
from copy import deepcopy


def generate_random_node(env_bounds):
    x_new = np.random.rand(2)
    x_new[0] = x_new[0]*(env_bounds[2]-env_bounds[0])+env_bounds[0]
    x_new[1] = x_new[1]*(env_bounds[3]-env_bounds[1])+env_bounds[1]
    return x_new


def convert_node_dict_to_node_array(node_attributes: Dict):
    n_nodes = len(list(node_attributes.items()))
    assert n_nodes, "Nothing in node_attributes"

    converter_key_to_array_index = {}
    _, v = list(node_attributes.items())[0]
    node_attr = np.zeros((n_nodes, v.shape[-1]))
    for i, (k, v) in enumerate(node_attributes.items()):
        converter_key_to_array_index[k] = i
        node_attr[i] = v

    converter_array_index_to_key = {v: k for k,
                                    v in converter_key_to_array_index.items()}
    return node_attr, converter_array_index_to_key


def steer(x_nearest, x_rand, steer_force=2):
    step = x_rand-x_nearest
    step = steer_force*step/np.linalg.norm(step)
    return x_nearest+step


def get_distances_to_point_coords(graph, point_coords, return_arr_graph_nodes_coords=False):
    graph_nodes_coords = nx.get_node_attributes(graph, "coords")
    arr_graph_nodes_coords, _ = convert_node_dict_to_node_array(
        graph_nodes_coords)

    cp_point_coords = deepcopy(point_coords)
    cp_point_coords = np.expand_dims(cp_point_coords, axis=0)
    cp_point_coords = np.ones(
        (arr_graph_nodes_coords.shape[0], 1))@cp_point_coords
    diff = arr_graph_nodes_coords-cp_point_coords
    dist = np.linalg.norm(diff, axis=-1)

    if return_arr_graph_nodes_coords:
        return dist, arr_graph_nodes_coords
    else:
        return dist


def get_id_nearest_node(graph, point_coords):
    dist = get_distances_to_point_coords(graph, point_coords)
    index_nearest = np.argmin(dist)
    return index_nearest


def get_near_nodes_id(graph, point_coords, min_radius, gamma=1):
    dist, graph_nodes_coords = get_distances_to_point_coords(
        graph, point_coords, return_arr_graph_nodes_coords=True)

    n_nodes = graph_nodes_coords.shape[0]
    r = np.min([(gamma*np.log(n_nodes)/n_nodes) **
               (1/graph_nodes_coords.shape[-1]), min_radius])
    near_nodes_id = np.where(dist <= r)[0]
    return near_nodes_id


def add_node_with_coords(graph, new_node_coords):
    id_new = graph.number_of_nodes()
    graph.add_node(id_new, coords=new_node_coords)
    return graph, id_new


def retrieve_path_from_tree_graph(tree_graph, id_end, id_start=0):
    path_edges = []
    id_curr = id_end
    senders = np.array([sender for sender, _, _ in tree_graph.edges.data()])
    receivers = np.array(
        [receiver for _, receiver, _ in tree_graph.edges.data()])
    while id_curr != id_start:
        # get predecessor
        id2edg_predecessors = np.where(receivers == id_curr)[0]
        predecessors = senders[id2edg_predecessors]
        assert predecessors.size <= 1, "Not a tree graph"
        path_edges.append((predecessors[0], id_curr))
        id_curr = predecessors[0]

    return path_edges[::-1]


def find_path_rrt_star(env, max_nodes, radius, start_node_coords, end_node_coords=None):
    env_bounds = env.bounds

    graph = nx.DiGraph()
    graph.add_node(0, coords=start_node_coords)
    node_costs = {0: 0}

    for iter_id in range(max_nodes):
        print(f"iteration n°{iter_id+1}/{max_nodes}", end="\r")
        # sample a random point
        x_rand = generate_random_node(env_bounds)
        id_nearest = get_id_nearest_node(graph, x_rand)
        x_nearest = nx.get_node_attributes(graph, "coords")[id_nearest]

        # avoid large steps, x_rand gives a direction more than anything
        x_new = steer(x_nearest, x_rand, steer_force=radius)

        if not (env.segments_are_in_an_object([[x_new, x_nearest]])):
            near_nodes_id = get_near_nodes_id(graph, x_new, radius)

            # add new node (after having taken the neighbours)
            graph, id_new = add_node_with_coords(graph, x_new)

            # find the best parent for the new node
            graph_nodes_coords = nx.get_node_attributes(graph, "coords")
            node_costs[id_new] = node_costs[id_nearest] + \
                np.linalg.norm(x_nearest-x_new)
            id_min = id_nearest

            for id_near in near_nodes_id:
                x_near = graph_nodes_coords[id_near]
                if not (env.segments_are_in_an_object([[x_near, x_new]])):
                    cand_cost = node_costs[id_near] + \
                        np.linalg.norm(x_near-x_new)
                    if cand_cost < node_costs[id_new]:
                        id_min = id_near
                        node_costs[id_new] = cand_cost

            graph.add_edge(id_min, id_new)

            # shortest path between two points is the segment...
            if (end_node_coords is not None) and not (env.segments_are_in_an_object([[x_new, end_node_coords]])):
                graph, id_end = add_node_with_coords(graph, end_node_coords)
                graph.add_edge(id_new, id_end)
                path_edges = retrieve_path_from_tree_graph(graph, id_end)
                return graph, path_edges

            # reorganize the tree graph
            for id_near in near_nodes_id:
                if id_near != id_min:
                    x_near = graph_nodes_coords[id_near]
                    if not (env.segments_are_in_an_object([[x_near, x_new]])) and node_costs[id_near] > node_costs[id_new]+np.linalg.norm(x_near-x_new):
                        # remove parent
                        parents = [
                            parent for parent in graph.predecessors(id_near)]
                        id_parent = parents[0]
                        graph.remove_edge(id_parent, id_near)

                        graph.add_edge(id_new, id_near)
    return graph, None


def color_path_and_graph_rrt_star(graph, path_edges):
    """
    Change the color in graph to show the path to take
    """
    path_nodes = [node0 for node0, _ in path_edges[1:]]
    extremity_nodes = [path_edges[0][0]]+[path_edges[-1][-1]]

    color_nodes = {}
    color_edges = {}
    for node_id, node_data in graph.nodes.data():
        color_nodes[node_id] = "#1f78b4"
        if node_id in path_nodes:
            color_nodes[node_id] = "#1fb451"
        if node_id in extremity_nodes:
            color_nodes[node_id] = "#e8b809"
    for sender, receiver, edge_data in graph.edges.data():
        edge_id = (sender, receiver)
        color_edges[edge_id] = "black"
        if edge_id in path_edges:
            color_edges[edge_id] = "#1fb451"

    nx.set_node_attributes(graph, color_nodes, "color")
    nx.set_edge_attributes(graph, color_edges, "color")


def plot_graph_with_path(graph, path_edges, title=""):
    color_path_and_graph_rrt_star(graph, path_edges)
    nx.draw_networkx(graph, pos=nx.get_node_attributes(graph, "coords"), node_size=5, with_labels=False, node_color=list(
        nx.get_node_attributes(graph, "color").values()), edge_color=list(nx.get_edge_attributes(graph, "color").values()), arrows=False)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    from course_2_path_finding.simple_env import SimpleEnv

    env_bounds = [0, 0, 100, 100]

    # Define a continuous environment with objects within
    simple_env = SimpleEnv(env_bounds)
    simple_env.add_disc(np.array([60, 85]), 35)
    simple_env.add_polygon(np.array([[10, 10], [10, 40], [40, 40], [40, 10]]))

    radius = 4
    max_nodes = 5000

    start_node_coords = np.array([0, 0])
    end_node_coords = np.array([100, 100])

    graph, path_edges = find_path_rrt_star(
        simple_env, max_nodes, radius, start_node_coords, end_node_coords)
    color_path_and_graph_rrt_star(graph, path_edges)
    simple_env.plot_objects(graph=graph, title="Example rrt* in use")
