
import numpy as np
#iner
from neuron import save_neuron_chains, get_neighborhood, rebuild_cities,init_neurons,get_routes,normalize
from distance import select_closest_gpid, routes_distances
import pandas as pd
from path import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

def SOM(args):
    """Solve the TSP using a Self-Organizing Map."""

    # Obtain the normalized set of cities (w/ coord in [0,1])
    cities = pd.read_csv(Path(args.data_dir) / 'data1.csv')

    iteration = args.iteration
    learning_rate = args.learning_rate
    decay = args.decay

    out_dir = Path(args.out_dir)
    out_dir.mkdir_p()

    cities_nm = cities.copy()

    cities_nm[['x', 'y']] = normalize(cities_nm[['x', 'y']])
    cities_nm.to_csv(out_dir/'cities_nm.csv')
    cities.to_csv(out_dir/'cities.csv')


    depot = cities_nm.query('city==0')[['x','y']].to_numpy()
    # The population size is 8 times the number of cities
    #n = cities_cp.shape[0] * 2# a single route's neurons
    n=100
    # Generate an adequate network of neurons:
    #network = generate_network(n)
    neuron_chains =init_neurons(size=n,depot=depot)
    print('--> Network of {} neurons created. Starting the iterations:'.format(n))
    best_routes=np.array([0])

    #save
    losses_sum_log={}#每个循环losses_sum值
    min_losses_sum_log = {}##保存最小值的路径losses
    min_losses_log={}#存储最好情况下四条路径的距离值
    min_routes_log={}
    best_id=0
    min_losses_sum=0

    for i in tqdm(range(iteration)):
        if not i % args.neuro_plot_freq:
            print('\t> Iteration {}/{}'.format(i, iteration), end="\r")
        # Choose a random city
        sample = cities_nm.sample(1)
        if int(sample['city']) in args.depot_idxs:
            continue
        city = sample[['x', 'y']].values#随机抽样 random  sampling
        group_idx,winner_idx = select_closest_gpid(neuron_chains, city)

        # Generate a filter that applies changes to the winner's gaussian
        gaussian = get_neighborhood(center=winner_idx, radix=n//10, domain=neuron_chains[0].shape[0])
        # Update the network's weights (closer to the city)
        neuron_chains[group_idx] += gaussian[:,np.newaxis] * learning_rate * (city - neuron_chains[group_idx])
        # Decay the variables
        learning_rate = learning_rate * decay
        n = n * decay


        if i % args.evaluate_freq==0:
            cities_od = rebuild_cities(cities_nm,neuron_chains,args.num_depots)
            cities_od[['x','y']] =cities.reindex(cities_od['city'])[['x','y']]
            losses = routes_distances(cities_od)
            losses_sum = sum(losses)
            losses_sum_log[i] = losses_sum

            if min_losses_sum == 0 or min_losses_sum > losses_sum:
                min_losses_sum = losses_sum
                best_id = i
                routes = get_routes(cities_od)
                routes = [list(item.astype(np.float64)) for item in routes]
                min_routes_log[i] = routes

                min_losses_sum_log[i] = losses_sum
                min_losses_log[i] = losses
                cities_od.to_csv(out_dir/'data_out_{:04d}.csv'.format(i))
                save_neuron_chains(neuron_chains,out_dir/"neuron_chains_{:04d}.npy".format(i))

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

    results = {}

    results['losses_sum_log']=losses_sum_log
    results['best_id'] = best_id

    results['min_losses_sum_log']=min_losses_sum_log
    results['min_losses_log']=min_losses_log
    results['min_routes_log'] = min_routes_log
    p = Path(out_dir/'results.json')

    with open(p, 'w') as fp:
        json.dump(results, fp)
        print('ok')


    return results