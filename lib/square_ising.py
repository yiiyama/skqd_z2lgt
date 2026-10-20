from collections import defaultdict
import rustworkx as rx


def make_primal_graph(nx, ny, charges):
    # Number of vertices
    nvx = nx + 1
    nvy = ny + 1

    primal_graph = rx.PyGraph()
    primal_graph.add_nodes_from([(ix, iy) for iy in range(nvy) for ix in range(nvx)])
    for iy in range(nvy):
        primal_graph.add_edges_from([(nvx * iy + ix, nvx * iy + ix + 1, -1.) for ix in range(nvx - 1)])
    for ix in range(nvx):
        primal_graph.add_edges_from([(nvx * iy + ix, nvx * (iy + 1) + ix, -1.) for iy in range(nvy - 1)])

    if charges:
        sx, sy = map(int, charges[0].split(','))
        tx, ty = map(int, charges[1].split(','))
        paths = rx.all_shortest_paths(primal_graph, nvx * sy + sx, nvx * ty + tx)
        if len(paths) != 1:
            raise ValueError('Shortest path is not unique between the charges')

        for vs, vt in zip(paths[0][:-1], paths[0][1:]):
            primal_graph.update_edge(vs, vt, 1.)

    return primal_graph


def get_hamiltonian_coeffs(primal_graph):
    nvx = max(datum[0] for datum in primal_graph.nodes()) + 1
    nvy = max(datum[1] for datum in primal_graph.nodes()) + 1
    nx = nvx - 1
    
    z_coeffs = defaultdict(float)
    zz_coeffs = defaultdict(float)
    for v1, v2, weight in primal_graph.edge_index_map().values():
        x1, y1 = primal_graph[v1]
        x2, y2 = primal_graph[v2]
        if x1 == x2:
            # Vertical primal link
            if x1 == 0:
                z_coeffs[y1 * nx] += weight
            elif x1 == nvx - 1:
                z_coeffs[(y1 + 1) * nx - 1] += weight
            else:
                zz_coeffs[(y1 * nx + x1 - 1, y1 * nx + x1)] += weight
        else:
            # Horizontal primal link
            if y1 == 0:
                z_coeffs[x1] += weight
            elif y1 == nvy - 1:
                z_coeffs[(y1 - 1) * nx + x1] += weight
            else:
                zz_coeffs[((y1 - 1) * nx + x1, y1 * nx + x1)] += weight

    return z_coeffs, zz_coeffs
