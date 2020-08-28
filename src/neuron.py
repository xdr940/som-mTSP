import numpy as np
import matplotlib.pyplot as plt
from distance import select_closest,select_closest_gpid

def generate_network(size):
    """
    Generate a neuron network of a given size.

    Return a vector of two dimensional points in the interval [0,1].
    """
    return np.random.rand(size, 2)

def init_neurons(size,depot):

    a = np.linspace([0.5,0.5],[0.1,1.],num=int(size/3))
    b = np.linspace([0.1,1.],[0.9,1.],num=int(size/3))
    c = np.linspace([0.9,1.],[0.5,0.5],num=int(size/3))
    u = np.concatenate([a,b,c],axis=0).copy()
    center = np.array([0.5,0.5])
    u-=center
    center-=center
    center = np.expand_dims(center,axis=0)
    u=np.concatenate([center,u],axis=0)

    r = np.rot90(u,1).transpose([1,0]).copy()


    l = (-r).copy()
    d = (-u).copy()

    l +=depot
    d += depot
    u += depot
    r += depot



    return [u,r,d,l]

def get_neighborhood(center, radix, domain):
    '''

    :param center: winner idx
    :param radix: 周围几个神经元
    :param domain: 总神经元个数
    :return:
    '''
    """Get the range gaussian of given radix around a center index."""

    # Impose an upper bound on the radix to prevent NaN and blocks
    if radix < 1:
        radix = 1

    # Compute the circular network distance to the center
    deltas = np.absolute(center - np.arange(domain))
    distances = np.minimum(deltas, domain - deltas)

    # Compute Gaussian distribution around the given center
    return np.exp(-(distances*distances) / (2*(radix*radix)))



def rebuild_cities(cities_nm, neuron_chains,num_depots):

    '''
    rebuild cities_nm
    :param cities:
    :param neuron_chains:
    :return:
    '''

    cities_od = cities_nm.copy()
    depots = cities_nm.head(num_depots)[['x','y']]
    gpids = -np.ones([len(cities_nm),2])
    gpids[num_depots:] = cities_od.iloc[num_depots:][['x', 'y']].apply(
        lambda c: select_closest_gpid(neuron_chains, c),
        axis=1, raw=True).to_numpy()
    idx = 0
    for chain,depot in zip(neuron_chains,depots.to_numpy()):
        gpids[:num_depots,0][idx] = idx
        gpids[:num_depots,1][idx] = select_closest(chain,depot)

        idx+=1


    cities_od['gid'] = gpids[:,0]
    cities_od['pid'] = gpids[:,1]

    cities_od = cities_od.sort_values(['gid', 'pid'],ascending=[True,True])

    return cities_od

def get_routes(cities_od):
    routes=[]
    for gid in range(int(cities_od['gid'].max())+1):
        routes.append(np.array(cities_od.query('gid=='+str(gid)).index))


    return routes
def save_neuron_chains(neuron_chains,path):
    neuron_chains_np = np.array(neuron_chains)
    np.save(path,neuron_chains_np)

def normalize(points):
    """
    Return the normalized version of a given vector of points.

    For a given array of n-dimensions, normalize each dimension by removing the
    initial offset and normalizing the points in a proportional interval: [0,1]
    on y, maintining the original ratio on x.
    """
    ratio = (points.x.max() - points.x.min()) / (points.y.max() - points.y.min()), 1
    ratio = np.array(ratio) / max(ratio)
    norm = points.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    return norm.apply(lambda p: ratio * p, axis=1)
