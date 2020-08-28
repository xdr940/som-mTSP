
import argparse
class OPT:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='M2CT2020')

        # -------------------------------
        self.parser.add_argument('--wk_root', type=str, default='/home/roit/aws/aprojects/M2CT2020/proj')
        self.parser.add_argument('--data_dir',default='./data')
        self.parser.add_argument('--iteration',default=10000)
        self.parser.add_argument('--neuro_plot_freq',default=10000)
        self.parser.add_argument('--evaluate_freq',default=100)

        self.parser.add_argument('--out_dir',default='./out_dir')

        self.parser.add_argument('--data_out',default='./data_out.csv')

        #mTSP args
        self.parser.add_argument('--depot_idxs',default=[0,1,2,3])
        self.parser.add_argument('--num_depots',default=4)
        self.parser.add_argument('--mode',
                                 choices=[
                                     'run',
                                     'visual'
                                 ],
                                 default='visual')

        # args
        self.parser.add_argument('--decay', default=0.99997)
        self.parser.add_argument('--learning_rate', default=0.99997)



    def args(self):
        self.options = self.parser.parse_args()
        return self.options

