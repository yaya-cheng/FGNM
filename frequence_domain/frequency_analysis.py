"""Implementation of sample attack."""
import os
from matplotlib import image
from numpy.testing._private.utils import requires_memory
import torch
import torchvision.models as models
from torch.autograd import Variable as V
from torch import nn
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import dct_my
from PIL import Image
import torch_dct
from utils import *
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
from Normalize import Normalize, TfNormalize
from loader import ImageNet
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
)

def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')

    model = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        TfNormalize('tensorflow'),
        net.KitModel(model_path).eval().cuda())
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--mean', type=float, default=np.array([0.485, 0.456, 0.406]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.229, 0.224, 0.225]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=10, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

transforms = T.Compose(
    [T.Resize(299),T.ToTensor()]
)


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def main():

    # model = torch.nn.Sequential(Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    #                             models.inception_v3(pretrained=True).cuda().eval())
    model = get_model('tf_inception_v3', '/mnt/hdd1/longyuyang/models_pt')

    # inc_v3 = torch.nn.Sequential(Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    #                             pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())
    # inc_v4 = torch.nn.Sequential(Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    #                             pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet').eval().cuda())


    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=10, shuffle=False, pin_memory=True, num_workers=8)
    total_grad = 0

    for images, images_ID,  gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda()
        images = images.cuda()
        N_grad= 0
        # for n in range(20):
        #     gauss = torch.randn(10, 3, 299, 299) * 0.05
        #     gauss = gauss.cuda()
        #     img_dct = dct_my.dct_2d(images + gauss)
        #     mask = (torch.rand_like(images) + 0.5).cuda()
        #     img_low = (img_dct * mask)
        #     img_low = V(img_low, requires_grad = True)
        #     img_idct = dct_my.idct_2d(img_low)
        #     output_v3 = model(img_idct)
        #     loss = F.cross_entropy(output_v3[0], gt+1)
        #     loss.backward()
        #     grad = img_low.grad.data
        #     grad = grad.mean(dim = 1).abs().mean(dim = 0).cpu().numpy()
        #     N_grad = N_grad + grad / 20.0
        # total_grad = total_grad + N_grad
        img_dct = dct_my.dct_2d(images)
        img_dct = V(img_dct, requires_grad = True)
        img_idct = dct_my.idct_2d(img_dct)
        output_v3 = model(img_idct)
        loss = F.cross_entropy(output_v3[0], gt+1)
        loss.backward()
        grad = img_dct.grad.data
        grad = grad.mean(dim = 1).abs().mean(dim = 0).cpu().numpy()
        total_grad += grad
    x = total_grad / 100.0
    x = (x - x.min()) / (x.max() - x.min())
    g1 = sns.heatmap(x, cmap="rainbow")
    g1.set(yticklabels=[])  # remove the tick labels
    g1.set(ylabel=None)  # remove the axis label
    g1.set(xticklabels=[])  # remove the tick labels
    g1.set(xlabel=None)  # remove the axis label
    g1.tick_params(left=False)
    g1.tick_params(bottom=False)
    sns.despine(left=True, bottom=True)
    plt.savefig('v3.png')



if __name__ == '__main__':
    main()