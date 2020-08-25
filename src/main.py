from sys import argv

import numpy as np
#iner
from neuron import generate_network, get_neighborhood, rebuild_cities,init_neurons,get_routes
from distance import select_closest_gpid, euclidean_distance, route_distance_p,route_distance,depot_with_chains,routes_distances
from plot import plot_neuron_chains, plot_route,plt_traj_p,plot_loss,plt_traj_np,plt_mtsp,plt_routes
from opts import OPT
from dataloader import dataloader
import pandas as pd
from path import Path
from tqdm import tqdm
from utils import normalize

import matplotlib.pyplot as plt


def run(args):


    df = pd.read_csv(Path(args.data_dir)/'data1.csv')
    result= som(df, args)
    #best_route, best_id, min_loss, losses, losses_decay
    #df_ordered = df.reindex(result['best_routes'])
    #distance = route_distance_p(args.data_out)

    if args.prt_route:
        print("--> best route and id".format(result['best_routes']),result['best_id'])
        print('--> Route found of length {}'.format(result['min_loss']))

    #if args.data_out:
    #    df_ordered.to_csv(args.data_out,index=False)

    return result['losses'],result['losses_decay']



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
    best_routes=np.array([0])
    best_id=0
    min_loss=0
    losses_log={}
    losses_decay = {}

    for i in tqdm(range(iteration)):
        if not i % args.neuro_plot_freq:
            print('\t> Iteration {}/{}'.format(i, iteration), end="\r")
        # Choose a random city
        sample = cities_nm.sample(1)
        if int(sample['city']) in args.depot_idxs:
            continue
        city = sample[['x', 'y']].values#随机抽样 random  sampling
        group_idx,winner_idx = select_closest_gpid(neuron_chains, city)
        #winner_idx_depot = depot_with_chains(neuron_chains,depot)

        # Generate a filter that applies changes to the winner's gaussian
        gaussian = get_neighborhood(center=winner_idx, radix=n//10, domain=neuron_chains[0].shape[0])
        # Update the network's weights (closer to the city)
        neuron_chains[group_idx] += gaussian[:,np.newaxis] * learning_rate * (city - neuron_chains[group_idx])
        # Decay the variables
        learning_rate = learning_rate * decay
        n = n * decay

        # Check for plotting interval
        #if not i % args._neuro_plot_freq:
            #plot_neuron_chains(cities_nm, neuron_chains)

        if i % args.evaluate_freq==0:
            #route = get_route(cities_cp, neuron_chains[group_idx])
            cities_od = rebuild_cities(cities_nm,neuron_chains,args.num_depots)
            routes = get_routes(cities_od)
            cities_od[['x','y']] =cities.reindex(cities_od['city'])[['x','y']]
            losses = routes_distances(cities_od)
            total_loss = sum(losses)

            if min_loss == 0 or min_loss > total_loss:
                min_loss = total_loss
                best_routes = routes
                best_id = i
                losses_decay[i] = total_loss
                cities_od.to_csv(out_dir/'data_out_{:04d}.csv'.format(i))
                if args.route_decay_log:
                    # 把渐渐缩小的最好路径存起来
                    # TODO
                    pass
            losses_log[i]=total_loss

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

    ret = {}


    ret['best_routes']=best_routes
    ret['best_id']=best_id

    ret['min_loss']=min_loss
    ret['losses']=losses_log
    ret['losses_decay']=losses_decay


    return ret


def main():
    pass
if __name__ == '__main__':
    args = OPT().args()
    #losses,losses_decay = run(args)
    #if args.plt_losses:
    #    plot_loss(losses, losses_decay)

    #plt_traj_p(args.data_out)
    #plt_traj_np(args)
    plt_routes(args.out_dir)


