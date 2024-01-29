from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


def closest_node(node, nodes):
    return nodes[cdist([node], nodes).argmin()]


def closest_n_points(nodes, N, meshXY):
    """
    Find the indices of the N closest points in a set of points.

    Parameters
    ----------
    nodes : np.ndarray
        The set of points to search.
    N : int
        The number of points to find.
    meshXY : np.ndarray
        The grid of points used for the KDTree.

    Returns
    -------
    np.ndarray
        The indices of the N closest points.

    """

    def do_kdtree(meshXY, points, N):
        mytree = cKDTree(meshXY)
        dist, indexes = mytree.query(points, range(1, N + 1))
        return indexes

    return do_kdtree(meshXY, nodes, N)
