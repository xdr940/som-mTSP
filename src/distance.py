import numpy as np
import pandas as pd
def select_closest(neurons, origin):
    """Return the index of the closest candidate to a given point."""
    return euclidean_distance(neurons, origin).argmin()


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

def depot_with_chains(neron_chains,depot):
    pass
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
    #修正depot点

    return [ret_gid,ret_pid]

def routes_distances(cities_od):
    distances=[]
    for gid in range(int(cities_od['gid'].max())+1):
        df = cities_od.query('gid=='+str(gid))
        points = df[['x', 'y']]
        distance = euclidean_distance(points, np.roll(points, 1, axis=0)).sum()
        distances.append(distance)

    return distances
