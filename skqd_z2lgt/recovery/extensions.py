"""State list extensions."""
from collections import defaultdict
import numpy as np
import jax
import jax.numpy as jnp
import rustworkx as rx
from heavyhex_qft.plaquette_dual import PlaquetteDual
from heavyhex_qft.elements import DummyPlaquette


def vertical_reflection(dual_lattice: PlaquetteDual):
    """Append a vertically reflected state for every entry in states.

    For this operation to make sense, the lattice must have an even number of plaquettes that can
    be divided into upper and lower halves, and the two subgraphs formed by those halves must be
    isomorphic to each other.
    """
    primal = dual_lattice.primal
    dual_graph = dual_lattice.graph
    num_p = dual_lattice.num_plaquettes
    # Split the graph into two halves
    upper = dual_graph.subgraph(list(range(num_p // 2)))
    lower = dual_graph.subgraph(list(range(num_p // 2, num_p)))
    # Identify plaquettes bordering the boundary
    boundary_matches = {}
    boundary = num_p // 2 - 0.5
    for n1, n2, _ in dual_graph.edge_index_map().values():
        plaq1 = dual_graph[n1]
        plaq2 = dual_graph[n2]
        if isinstance(plaq1, DummyPlaquette) or isinstance(plaq2, DummyPlaquette):
            continue
        if (((p1 := dual_graph[n1]) - boundary) * ((p2 := dual_graph[n2]) - boundary)) < 0:
            boundary_matches[p1] = p2
            boundary_matches[p2] = p1

    # Node matcher function. Anchor the two subgraphs at the boundary.
    def node_matcher(p1, p2):
        if (match := boundary_matches.get(p1)) is None:
            return True
        return p2 == match

    vf2 = list(rx.vf2_mapping(upper, lower, node_matcher=node_matcher))[0]
    # Indices into the last dim of the state array
    # The actual packed state array has a padding bit, so we add 1 to the indices
    upper_indices = [primal.plaquette_id_to_idx(upper[key]) + 1 for key in vf2.keys()]
    lower_indices = [primal.plaquette_id_to_idx(lower[value]) + 1 for value in vf2.values()]

    # We can now permute the states array with packbits(unpackbits(states_p)[:, permutation]) but
    # that incurs a huge memory cost. Instead we work directly on the bits of states_p below.
    # Group the permutations by actual bitshift actions
    # (out_byte, in_byte, left_shift) -> mask on the source byte
    groups = defaultdict(int)
    for out_bit, in_bit in zip(upper_indices + lower_indices, lower_indices + upper_indices):
        ob, ob_pos = divmod(int(out_bit), 8)
        ib, ib_pos = divmod(int(in_bit), 8)
        # Big-endian: bit k of a byte carries the value 1 << (7 - k)
        groups[(ob, ib, ib_pos - ob_pos)] |= 1 << (7 - ib_pos)

    # Make the bitshift action plan
    num_out_bytes = -(-(num_p + 1) // 8)
    plan = [[] for _ in range(num_out_bytes)]
    for (ob, ib, shift), mask in sorted(groups.items()):
        plan[ob].append((ib, shift, np.uint8(mask)))

    def extend(states_p: jax.Array) -> jax.Array:
        """Extend the states array with vertically reflected plaquette configurations.
        
        The bits are moved directly within the packed representation: for every (output byte,
        input byte, shift) combination, the relevant bits are selected with a mask and shifted in a
        single operation. The plan is built on the host from the concrete indices, so the compiled
        kernel is a single fused elementwise pass over the input (no (N, num_bits) intermediate).
        Bit order is big-endian, as in jnp.unpackbits / jnp.packbits defaults.
        """
        out = []
        for terms in plan:
            acc = jnp.zeros(states_p.shape[:-1], dtype=jnp.uint8)
            for ib, shift, mask in terms:
                term = states_p[..., ib] & mask
                if shift > 0:
                    term = term << shift
                elif shift < 0:
                    term = term >> -shift
                acc = acc | term
            out.append(acc)
        reflected = jnp.stack(out, axis=-1)
        return jnp.concatenate([states_p, reflected], axis=0)

    return extend
