import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from path import Path
from distance import route_distance

def plot_neuron_chains(df,chains):
    x = np.array(df['x'])
    y = np.array(df['y'])

    x_ = np.expand_dims(x, axis=1)
    y_ = np.expand_dims(y, axis=1)

    xy = np.concatenate([x_, y_], axis=1)
    plt.close()
    plt.figure()
    #绘点
    plt.scatter(x, y, c='blue')
    start = df.query('city==0')

    plt.scatter(float(start['x']), float(start['y']), c='red')
    # 点标号

    for index, row in df.iterrows():
        plt.text(row['x'], row['y'], int(row['city']), fontsize=10)
    color=['r','g','b','k']
    for idx,chain in enumerate(chains):
        plt.plot(chain[:,0],chain[:,1],color[idx])


    pass

def plot_network(cities, neurons, name, ax=None):
    """Plot a graphical representation of the problem"""
    mpl.rcParams['agg.path.chunksize'] = 10000

    if not ax:
        fig = plt.figure(figsize=(5, 5), frameon = False)
        axis = fig.add_axes([0,0,1,1])

        axis.set_aspect('equal', adjustable='datalim')
        plt.axis('off')

        axis.scatter(cities['x'], cities['y'], color='red', s=4)
        axis.plot(neurons[:,0], neurons[:,1], 'r.', ls='-', color='#0063ba', markersize=2)

        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()

    else:
        ax.scatter(cities['x'], cities['y'], color='red', s=4)
        ax.plot(neurons[:,0], neurons[:,1], 'r.', ls='-', color='#0063ba', markersize=2)
        return ax

def plot_route(cities, route, name, ax=None):
    """Plot a graphical representation of the route obtained"""
    mpl.rcParams['agg.path.chunksize'] = 10000

    if not ax:
        fig = plt.figure(figsize=(5, 5), frameon = False)
        axis = fig.add_axes([0,0,1,1])

        axis.set_aspect('equal', adjustable='datalim')
        plt.axis('off')

        axis.scatter(cities['x'], cities['y'], color='red', s=4)
        route = cities.reindex(route)
        route.loc[route.shape[0]] = route.iloc[0]
        axis.plot(route['x'], route['y'], color='purple', linewidth=1)

        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()

    else:
        ax.scatter(cities['x'], cities['y'], color='red', s=4)
        route = cities.reindex(route)
        route.loc[route.shape[0]] = route.iloc[0]
        ax.plot(route['x'], route['y'], color='purple', linewidth=1)
        return ax


def plt_traj(df,array):
    x = np.array(df['x'])
    y =np.array( df['y'])


    x_ = np.expand_dims(x,axis=1)
    y_ = np.expand_dims(y,axis=1)

    xy = np.concatenate([x_,y_],axis=1)


    #点标号
    for index,row in df.iterrows():
        plt.text(row['x'],row['y'],int(row['city']),fontsize=10)

    print(df['city'].to_numpy())



    #绘点
    plt.scatter(x,y,c='blue')
    start = df.query('city==0')

    plt.scatter(float(start['x']),float(start['y']),c='red')


    # 画路径
    plt.plot(x, y, 'r')
    plt.plot([x[0], x[-1]], [y[0], y[-1]], 'r')
    #plt.xlim([169, 171])
    #plt.ylim([35, 37])

    plt.show()

def plot_loss(losses,losses_decay):
    plt.figure()
    plt.plot(losses.keys(),losses.values())
    plt.plot(losses_decay.keys(),losses_decay.values(),'r-o')
    plt.legend(['losses','min_loss'])
    plt.title('Distance of Routes')
    plt.xlabel('Iterations')
    plt.ylabel('Distance')



def plt_traj_p(path):
    df = pd.read_csv(path)
    x = np.array(df['x'])
    y =np.array( df['y'])


    x_ = np.expand_dims(x,axis=1)
    y_ = np.expand_dims(y,axis=1)


    #点标号
    for index,row in df.iterrows():
        plt.text(row['x'],row['y'],int(row['city']),fontsize=10)

    print(df['city'].to_numpy())
    #画路径
    plt.plot(x,y,'r')
    plt.plot([x[0],x[-1]],[y[0],y[-1]],'r')


    #绘点
    plt.scatter(x,y,c='blue')
    start = df.query('city==0')

    plt.scatter(float(start['x']),float(start['y']),c='red')

    #plt.xlim([169, 171])
    #plt.ylim([35, 37])
    plt.show()

    pass



def plt_traj_np(args):
    p = Path(args.data_dir)/'data1.csv'
    df = pd.read_csv(p)
    route = np.array(args.route_plt)
    df =df.reindex(route)
    dis = route_distance(df)
    print("--> route distance:{}".format(dis))
    x = np.array(df['x'])
    y =np.array( df['y'])




    #点标号
    for index,row in df.iterrows():
        plt.text(row['x'],row['y'],int(row['city']),fontsize=10)

    print(df['city'].to_numpy())
    #画路径
    plt.plot(x,y,'r')
    plt.plot([x[0],x[-1]],[y[0],y[-1]],'r')


    #绘点
    plt.scatter(x,y,c='blue')
    start = df.query('city==0')
    plt.title('Best Route')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.scatter(float(start['x']),float(start['y']),c='red')

    #plt.xlim([169, 171])
    #plt.ylim([35, 37])

    plt.show()

    pass


def plt_mtsp(args):
    p = Path(args.data_dir) / 'data1.csv'
    df = pd.read_csv(p)

    # 点标号
    for index, row in df.iterrows():
        plt.text(row['x'], row['y'], int(row['city']), fontsize=10)
    print(df['city'].to_numpy())

    x=df['x']
    y=df['y']
    # 绘点
    plt.scatter(x, y, c='blue')
    start = df.query('city==0')
    plt.title('Best Routes')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.scatter(float(start['x']), float(start['y']), c='red')

    routes = args.routes

    routes_nps = []
    for route in routes:
        route = np.array(route)
        routes_nps.append(route)
    dfs=[]
    dises=[]
    xs=[]
    ys=[]
    colors=['r','g','b','k']
    for idx,item in enumerate(routes_nps):
        df_ = df.reindex(item)
        dfs.append(df_)
        dises.append(route_distance(df_))
        print("--> route distance:{}".format(dises[idx]))
        xs.append(np.array(dfs[-1]['x']))
        ys.append(np.array(dfs[-1]['y']))
        plt.plot(xs[-1],ys[-1],colors[idx])
        plt.plot([xs[-1][0],xs[-1][-1]],[ys[-1][0],ys[-1][-1]],colors[idx])

    print(np.array(dises).sum())





    # plt.xlim([169, 171])
    # plt.ylim([35, 37])

    plt.show()

def plt_routes(dir):
    plt.figure()
    files = Path(dir).files('*.csv')
    files.sort()
    path = files[-1]
    df = pd.read_csv(path)

    x = np.array(df['x'])
    y = np.array(df['y'])


    # 点标号
    for index, row in df.iterrows():
        plt.text(row['x'], row['y'], int(row['city']), fontsize=10)

    print(df['city'].to_numpy())

    # 绘点
    plt.scatter(x, y, c='blue')
    start = df.query('city==0')

    plt.scatter(float(start['x']), float(start['y']), c='red')

    groups = df['gid'].max()
    colors=['r','g','b','k']
    for gid in range(int(groups)+1):
        x = df.query('gid=='+str(gid))['x'].to_numpy()
        y = df.query('gid=='+str(gid))['y'].to_numpy()
    # 画路径
        plt.plot(x, y, colors[gid])
        plt.plot([x[0], x[-1]], [y[0], y[-1]], colors[gid])
    plt.title(path.stem)

    plt.show()