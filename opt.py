import argparse


parser = argparse.ArgumentParser()

# train schedule
parser.add_argument('-epochs', type=int, default=4000, help='total epochs to train')
parser.add_argument('-save_ckp', type=int, default=400, help='save ckp every 400 steps')

# lr schedule
parser.add_argument('-lr', type=float, default=1e-4, help='learning rate of a multi-layer perception')
parser.add_argument('-lr_decay_ep', type=int, default=1333, help='the number of epochs for decay learning rate')
parser.add_argument('-lr_decay_rate', type=float, default=0.1, help='ratio of decay rate every <lr_decay_ep> epochs')

# net work
parser.add_argument('-bs', type=int, default=2048, help='data loader batch size')
parser.add_argument('-hidden', type=int, default=256, help='the number of hidden node of the main MLP')
parser.add_argument('-img_fea_ch', type=int, default=256, help='the number of img feature channel by wavelet transform')
parser.add_argument('-mlp_layer', type=int, default=8, help='the number of mlp hidden layers')

# training and eval data
parser.add_argument('-scene', type=str, default="", help='directory to the datasets dir')
# train opt
parser.add_argument('-render_viewing', action='store_true', help='generate view-dependent-effect video')
# eval opt
# parser.add_argument('-eval_path', type=str, default='runs/evaluation/', help='path to save validation image')

parser.add_argument('-model_dir', type=str, default="", help='load model ckp directory')
parser.add_argument('-restart', action='store_true', help='delete old weight and retrain')
parser.add_argument('-clean', action='store_true', help='delete old weight without start training process')

args = parser.parse_args()





