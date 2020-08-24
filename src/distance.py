import numpy as np
import pandas as pd
def select_closest(candidates, origin):
    """Return the index of the closest candidate to a given point."""
    return euclidean_distance(candidates, origin).argmin()


def euclidean_distance(a, b):
    """Return the array of distances of two numpy arrays of points."""
    temp = a-b
    return np.linalg.norm(temp, axis=1)

def route_distance(cities):
    """Return the cost of traversing a route of cities in a certain order."""
    points = cities[['x', 'y']]
    distances = euclidean_distance(points, np.roll(points, 1, axis=0))
    return np.sum(distances)


def route_distance_p(path):
    """Return the cost of traversing a route of cities in a certain order."""
    cities = pd.read_csv(path)
    points = cities[['x', 'y']]
    distances = euclidean_distance(points, np.roll(points, 1, axis=0))
    return np.sum(distances)


def select_closest_gpid(neuron_chains, origin):
    ret_gid=0
    ret_pid=0
    min_value=100
    for gid,chain in enumerate(neuron_chains):
        dises = euclidean_distance(chain, origin)
        value = dises.min()
        idx = dises.argmin()
        if min_value>value:
            min_value = value
            ret_gid = gid
            ret_pid = idx

    return [ret_gid,ret_pid]

def routes_distances(cities):
    points = cities.query('gid==0').sort_values('pid')[['x', 'y']]
    distances = euclidean_distance(points, np.roll(points, 1, axis=0))
    print(distances)
    pass

