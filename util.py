import random

import igraph as ig
import jax
import jax.numpy as jnp


def scale_free_graph(rng: jax.random.PRNGKeyArray,
                     n_vars: int,
                     edges_per_var: int = 2,
                     power: float = 1.0,
                     transpose: bool = True) -> jax.Array:
    """ Generate scale-free graph with given number of nodes and edges per node.
        Also known as Barabasi-Albert model.
        See https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model
        for more details.

    Args:
        rng (jax.random.PRNGKeyArray): random generator key for jax
        n_vars (int): number of nodes
        edges_per_var (int): number of edges per node, in expectation
        power (float, optional): power in preferential attachment process. 
         Higher values make few nodes have high out-degree. Defaults to 1.0.
        transpose (bool, optional): whether to transpose the adjacency matrix. Defaults to True.
            NOTE: transposing the matrix corresponds to a power-law out-degree.
            This is the default, as it corresponds to findings that real gene-regulatory networks
            have power-law out-degree distributions.

    Returns:
        np.ndarray: adjacency matrix of the generated graph
    """
    rng, sample_rng = jax.random.split(rng)
    perm = jax.random.permutation(sample_rng, n_vars).tolist()
    # need to seed random for ig graph
    random.seed(jax.random.randint(rng, shape=(), minval=0, maxval=2**31 - 1).item())
    g = ig.Graph.Barabasi(n=n_vars,
                          m=edges_per_var,
                          directed=True,
                          power=power).permute_vertices(perm)
    mat = jnp.array(g.get_adjacency().data).astype(int)
    if transpose:
        mat = mat.T
    return mat