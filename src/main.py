from som import SOM
from plot import plot_neuron_chains,plot_loss,plot_routes
from opts import OPT






if __name__ == '__main__':
    args = OPT().args()
    if args.mode=='run':
        SOM(args)
    else:
        plot_loss(input_dir = args.out_dir)
        plot_routes(input_dir = args.out_dir)
        plot_neuron_chains(input_dir = args.out_dir)


