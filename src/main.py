from sys import argv

import numpy as np
#iner
from neuron import generate_network, get_neighborhood, rebuild_cities,init_neurons,get_routes
from distance import select_closest_gpid, euclidean_distance, route_distance_p,route_distance,routes_distances
from plot import plot_neuron_chains, plot_route,plt_traj_p,plot_loss,plt_traj_np,plt_mtsp
from opts import OPT
from dataloader import dataloader
import pandas as pd
from path import Path
from tqdm import tqdm
from utils import normalize

import matplotlib.pyplot as plt


def run(args):


    df = pd.read_csv(Path(args.data_dir)/'data1.csv')
    best_route,best_id,min_loss,losses,losses_decay = som(df, args)
    df_ordered = df.reindex(best_route)
    distance = route_distance_p(args.data_out)

    if args.prt_route:
        print("--> best route and id".format(np.array(best_route)),best_id)
        print('--> Route found of length {}'.format(distance))

    if args.data_out:
        df_ordered.to_csv(args.data_out,index=False)

    return losses,losses_decay



def som(cities, args):
    """Solve the TSP using a Self-Organizing Map."""

    # Obtain the normalized set of cities (w/ coord in [0,1])
    iteration = args.iteration
    learning_rate = args.learning_rate
    decay = args.decay

    out_dir = Path(args.out_dir)
    out_dir.mkdir_p()

    cities_nm = cities.copy()

    cities_nm[['x', 'y']] = normalize(cities_nm[['x', 'y']])

    depot = cities_nm.query('city==0')[['x','y']].to_numpy()
    # The population size is 8 times the number of cities
    #n = cities_cp.shape[0] * 2# a single route's neurons
    n=100
    # Generate an adequate network of neurons:
    #network = generate_network(n)
    neuron_chains =init_neurons(size=n,depot=depot)
    print('--> Network of {} neurons created. Starting the iterations:'.format(n))
    best_route=np.array([0])
    best_id=0
    min_loss=0
    losses={}
    losses_decay = {}

    for i in tqdm(range(iteration)):
        if not i % args._neuro_plot_freq:
            print('\t> Iteration {}/{}'.format(i, iteration), end="\r")
        # Choose a random city
        city = cities_nm.sample(1)[['x', 'y']].values#随机抽样 random  sampling
        group_idx,winner_idx = select_closest_gpid(neuron_chains, city)
        # Generate a filter that applies changes to the winner's gaussian
        gaussian = get_neighborhood(center=winner_idx, radix=n//10, domain=n)
        # Update the network's weights (closer to the city)
        neuron_chains[group_idx] += gaussian[:,np.newaxis] * learning_rate * (city - neuron_chains[group_idx])
        # Decay the variables
        learning_rate = learning_rate * decay
        n = n * decay

        # Check for plotting interval
        #if not i % args._neuro_plot_freq:
        plot_neuron_chains(cities_nm, neuron_chains)

        if i % args.evaluate_freq==0:
            #route = get_route(cities_cp, neuron_chains[group_idx])
            cities_od = rebuild_cities(cities_nm,neuron_chains)
            routes = get_routes(cities_od)

            loss = routes_distances(cities_nm)
            if  min_loss==0 or min_loss > loss:
                min_loss=loss
                best_route = route
                best_id = i
                losses_decay[i] = loss
                if args.route_decay_log:
                    #把渐渐缩小的最好路径存起来
                    #TODO
                    pass
            losses[i] = loss
    #end for

        # Check if any parameter has completely decayed.
        if n < 1:
            print('Radius has completely decayed, finishing execution',
            'at {} iterations'.format(i))
            break
        if learning_rate < 0.001:
            print('Learning rate has completely decayed, finishing execution',
            'at {} iterations'.format(i))
            break


    print('Completed {} iterations.'.format(iteration))

    #plot_network(cities, network, name=out_dir/'final.png')

    #route = get_route(cities, network)

    #plot_route(cities, route, out_dir/'route.png')
    return best_route,best_id,min_loss,losses,losses_decay


def main():
    pass
if __name__ == '__main__':
    args = OPT().args()
    losses,losses_decay = run(args)
    if args.plt_losses:
        plot_loss(losses, losses_decay)
    plt.figure()
    plt_traj_p(args.data_out)
    plt_traj_np(args)

