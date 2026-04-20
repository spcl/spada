from typing import List
import numpy as np
import igraph


def linearize_with_random_forest(g: igraph.Graph, order: List[int], base_size: int = 4) -> None:
    """
    Pushes the original_id's of the vertices of g onto the order, attempting to minimize the cost of the linear arrangement.
    The cost of the linear arrangement is given by summing the distance of the edge's endoints in the order over all edges.
    :param g: a graph
    :param order: a list of vertex ids
    :param base_size: non-negative integer that controls on which size connected components the linear arrangement switches to
    a naive method. You should set this base_size smaller than the desired bandwidth. It is recommended to use a value
    larger than 1 to improve performance.
    :return:
    """
    weights = np.random.rand(g.ecount())
    spanning_forest = g.spanning_tree(weights=weights, return_tree=True)

    assert spanning_forest.vcount() == g.vcount()

    components = spanning_forest.connected_components(mode='weak')

    assert spanning_forest.ecount() == spanning_forest.vcount() - len(components)

    for cc in components:

        # For very small components, do not bother to optimize as the bandwidth is bounded by the size of the component
        if len(cc) <= base_size:
            order.extend(g.vs[cc]["original_id"])
            continue

        g_cc = spanning_forest.subgraph(cc)

        dfs_v, dfs_p = g_cc.dfs(0, mode='all')

        n = len(cc)
        assert len(dfs_v) == n

        tree_directed = igraph.Graph(n=n, edges=zip(dfs_p[1:], dfs_v[1:]), directed=True)
        tree_directed.vs["original_id"] = g_cc.vs["original_id"]

        assert tree_directed.ecount() == tree_directed.vcount( ) -1
        assert tree_directed.vcount() == n

        linearize_tree(tree_directed, order)


def linearize_with_ck(g: igraph.Graph, order: List[int], base_size = 2)-> None:

    components = g.connected_components(mode='weak')

    for cc in components:
        # For very small components, do not bother to optimize as the bandwidth is bounded by the size of the component
        if len(cc) <= base_size:
            order.extend(g.vs[cc]["original_id"])
            continue

        g_cc = g.subgraph(cc)

        it = g_cc.bfsiter(0, mode='all')
        for v in it:
            order.append(v["original_id"])
            continue


def linearize_tree(g: igraph.Graph, order: List[int]) -> None:
    """
    Pushes the original_id's of the vertices of g onto the order, attempting to minimize the cost of the linear arrangment.
    The cost of the linear arrangement is given by summing the distance of the edge's endoints in the order over all edges.
    :param g: A directed, rooted tree
    :param order: a list of vertex ids
    :return: None
    """

    # This is a little mini DP that computes the size of the subtrees
    # We compute the subtree sizes in reverse topological order
    topo = g.topological_sorting()
    g.vs["subtree_size"] = 1
    for v in reversed(topo):
        for u in g.successors(v):
            g.vs[v]["subtree_size"] = g.vs[v]["subtree_size"] + g.vs[u]["subtree_size"]

    assert g.vs[topo[0]]["subtree_size"] == g.vcount()

    _linearize_tree_stack(g, g.vs[topo[0]], order)


def _linearize_tree_stack(g: igraph.Graph, root, order: List[int]) -> None:

    stack = [root]
    while len(stack) > 0:

        current_vertex = stack.pop()
        order.append(current_vertex["original_id"])

        # We want to visit the larger subtree vertices last, hence we push onto the stack into reverse order
        succ = g.vs[g.successors(current_vertex)]
        children = sorted(succ, key=lambda k: k["subtree_size"], reverse=True)
        stack.extend(children)


def _linearize_tree_recursively(g: igraph.Graph, root, order: List[int]) -> None:

    order.append(root["original_id"])
    succ = g.vs[g.successors(root)]
    next_subtree = sorted(succ, key=lambda k: k["subtree_size"])
    for u in next_subtree:
        _linearize_tree_recursively(g, u, order)