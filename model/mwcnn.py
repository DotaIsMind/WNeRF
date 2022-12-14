from model import common
import torch
import torch.nn as nn
import scipy.io as sio

def make_model(args, parent=False):
    return MWCNN(args)

class MWCNN(nn.Module):
    # def __init__(self, args, conv=common.default_conv):
    def __init__(self, conv=common.default_conv):
        super(MWCNN, self).__init__()
        # n_resblocks = args.n_resblocks
        # default is 20
        n_resblocks = 20

        # n_feats = args.n_feats
        # default is 64, but NeRF
        n_feats = 64
        kernel_size = 3
        self.scale_idx = 0
        # default is 3
        # nColor = args.n_colors
        nColor = 3

        act = nn.ReLU(True)

        self.DWT = common.DWT()
        # self.IWT = common.IWT()

        n = 1
        m_head = [common.BBlock(conv, nColor, n_feats, kernel_size, act=act)]
        d_l0 = []
        d_l0.append(common.DBlock_com1(conv, n_feats, n_feats, kernel_size, act=act, bn=False))


        d_l1 = [common.BBlock(conv, n_feats * 4, n_feats * 2, kernel_size, act=act, bn=False)]
        d_l1.append(common.DBlock_com1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False))

        d_l2 = []
        d_l2.append(common.BBlock(conv, n_feats * 8, n_feats * 4, kernel_size, act=act, bn=False))
        d_l2.append(common.DBlock_com1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False))
        pro_l3 = []
        pro_l3.append(common.BBlock(conv, n_feats * 16, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(common.DBlock_com(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(common.DBlock_inv(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(common.BBlock(conv, n_feats * 8, n_feats * 16, kernel_size, act=act, bn=False))

        # i_l2 = [common.DBlock_inv1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False)]
        # i_l2.append(common.BBlock(conv, n_feats * 4, n_feats * 8, kernel_size, act=act, bn=False))
        #
        # i_l1 = [common.DBlock_inv1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False)]
        # i_l1.append(common.BBlock(conv, n_feats * 2, n_feats * 4, kernel_size, act=act, bn=False))
        #
        # i_l0 = [common.DBlock_inv1(conv, n_feats, n_feats, kernel_size, act=act, bn=False)]
        #
        # m_tail = [conv(n_feats, nColor, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l0 = nn.Sequential(*d_l0)
        self.pro_l3 = nn.Sequential(*pro_l3)
        # self.i_l2 = nn.Sequential(*i_l2)
        # self.i_l1 = nn.Sequential(*i_l1)
        # self.i_l0 = nn.Sequential(*i_l0)
        # self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # [bs, 3, 100, 100]=>[bs, 64, 100, 100]
        x0 = self.d_l0(self.head(x))
        # [bs, 64, 100, 100]=>[bs, 128, 50, 50]
        x1 = self.d_l1(self.DWT(x0))
        # [bs, 127, 50, 50]=>[bs, 256, 25, 25]
        x2 = self.d_l2(self.DWT(x1))
        # out = self.pro_l3(self.DWT(x2))
        # x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        # x_ = self.IWT(self.i_l2(x_)) + x1
        # x_ = self.IWT(self.i_l1(x_)) + x0
        # x = self.tail(self.i_l0(x_)) + x

        return x2

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


if __name__ == "__main__":
    import numpy as np
    model = MWCNN()
    # print(model)
    data = np.load("../../tiny_nerf_data.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = data['images']
    poses = data['poses']
    focal = data['focal']

    # ??????????????????
    k = 3
    train_list = np.arange(images.shape[0])
    np.random.shuffle(train_list)
    train_list = train_list[:k]

    H, W = images.shape[1:3]

    train_images = torch.tensor(images[train_list]).permute(0, 3, 1, 2)
    with torch.no_grad():
        reference_feature = model(train_images)
