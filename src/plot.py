import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from path import Path
import json
from tqdm import tqdm
from distance import route_distance

def plot_neuron_chains(input_dir):

    input_dir = Path(input_dir)

    dump_dir = input_dir / 'fig'
    dump_dir.mkdir_p()

    df = pd.read_csv(input_dir/'cities_nm.csv')


    def back_grd():




        x = np.array(df['x'])
        y = np.array(df['y'])

        x_ = np.expand_dims(x, axis=1)
        y_ = np.expand_dims(y, axis=1)

        plt.close()
        plt.figure(1)
        #绘点
        plt.scatter(x, y, c='blue')
        start = df.query('city==0')

        plt.scatter(float(start['x']), float(start['y']), c='red')
        # 点标号

        for index, row in df.iterrows():
            plt.text(row['x'], row['y'], int(row['city']), fontsize=10)
    color=['r','g','b','k']


    #神经元绘制
    chain_files = input_dir.files('*.npy')
    chain_files.sort()
    axis=[]
    plt.close()
    plt.figure()
    for file in chain_files:
        chains = np.load(file)
        back_grd()
        for idx,chain in enumerate(chains):
            axis.append(plt.plot(chain[:,0],chain[:,1],color[idx]))
        plt.title(file.stem)
        plt.savefig(dump_dir/(file.stem+'.png'), bbox_inches='tight', pad_inches=0, dpi=200)
        plt.clf()
    plt.close()
def plot_loss(input_dir):
    input_dir = Path(input_dir)
    input_dir.mkdir_p()
    json_file = input_dir/'results.json'
    with open(json_file, encoding='utf-8') as f:
        content = f.read()
        results = json.loads(content)
    min_losses_sum_log = results['min_losses_sum_log']
    losses_sum_log = results['losses_sum_log']

    min_losses_sum_log_x = list(min_losses_sum_log.keys())
    min_losses_sum_log_x = [float(item) for item in min_losses_sum_log_x]
    min_losses_sum_log_y = list(min_losses_sum_log.values())
    min_losses_sum_log_y = [float(item) for item in min_losses_sum_log_y]
    losses_sum_log_x = list(losses_sum_log.keys())
    losses_sum_log_x = [float(item) for item in losses_sum_log_x]
    losses_sum_log_y = list(losses_sum_log.values())
    losses_sum_log_y = [float(item) for item in losses_sum_log_y]

    plt.close()
    plt.figure()
    plt.plot(losses_sum_log_x,losses_sum_log_y)
    plt.plot(min_losses_sum_log_x,min_losses_sum_log_y,'r-o')
    plt.legend(['losses','min_loss'])
    plt.title('Distance of Routes')
    plt.xlabel('Iterations')
    plt.ylabel('Distance')

    dump_dir = input_dir / 'fig'
    dump_dir.mkdir_p()

    plt.savefig(dump_dir/'losses_sum.png',bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()

    #
    print('-----------plot_loss--------------')
    best_id = str(results['best_id'])
    best_routes = results['min_routes_log'][best_id]
    min_losses = results['min_losses_log'][best_id]
    min_losses_sum = results['min_losses_sum_log'][best_id]

    print('--> best routes and distances')
    for route,loss in zip(best_routes,min_losses):
        print('route:')
        print([int(item) for item in route])
        print('distances:{}\n'.format(loss))
    print('--> min distances sum: {}'.format(min_losses_sum))





def plot_routes(input_dir):
    def back_ground():
        df = pd.read_csv(input_dir/'cities.csv')
        x = np.array(df['x'])
        y = np.array(df['y'])

        plt.figure()
        # 点标号
        for index, row in df.iterrows():
            plt.text(row['x'], row['y'], int(row['city']), fontsize=10)
        # 绘点
        plt.scatter(x, y, c='blue')
        start = df.query('city==0')
        plt.scatter(float(start['x']), float(start['y']), c='red')


    input_dir = Path(input_dir)
    files = input_dir.files('*.csv')
    files.sort()





    files = input_dir.files('*.csv')
    files.sort()
    dump_dir = input_dir / 'fig'
    dump_dir.mkdir_p()
    plt.close()
    plt.figure()
    for file in tqdm(files):
        if file.stem in['cities_nm','cities']:
            continue
        back_ground()

        df = pd.read_csv(file)
        groups = df['gid'].max()
        colors=['r','g','b','k']
        for gid in range(int(groups)+1):
            x = df.query('gid=='+str(gid))['x'].to_numpy()
            y = df.query('gid=='+str(gid))['y'].to_numpy()
        # 画路径
            plt.plot(x, y, colors[gid])
            plt.plot([x[0], x[-1]], [y[0], y[-1]], colors[gid])
        plt.title(file.stem)
        plt.savefig(dump_dir/(file.stem+'.png'),bbox_inches='tight', pad_inches=0, dpi=200)
        plt.clf()
    plt.close()

