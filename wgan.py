import matplotlib.pyplot as plt
from os import listdir
from PIL import Image as PImage
import numpy as np
import os
import torchvision
from torchvision.utils import save_image
import torch
import pandas as pd
from fastai.vision import *
from fastai.vision.gan import *
from fastai.utils.mem import *
from fastai.callbacks import *


def train_wgan (data, epochs):
    
    #create generator and critic
    generator = basic_generator(in_size=128, n_channels=3, n_extra_layers=1)
    critic    = basic_critic   (in_size=128, n_channels=3, n_extra_layers=1)
    
    #create GAN learner
    learn = GANLearner.wgan(data, generator, critic, switch_eval=False,
                        opt_func = partial(optim.Adam, betas = (0., 0.99)), wd=0.)
    
    #trainModel
    learn.fit(epochs ,2e-2)
    return generator


def get_data_mask (path, df, bs=128, size=128):

    return (GANItemList.from_df(path=path, df=df, noise_sz=100)
               .split_none()
               .label_from_func(noop)
               .transform(tfms=None, size=size, tfm_y=True)
               .databunch(bs=bs)
               .normalize(stats = [torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])], do_x=False, do_y=True))


def generate_masks(file_names, num_data_gen, path_mask_real, epochs_train=200, bs=128, size=128):
    
    data = get_data_mask(path_mask_real, file_names, bs, size)
    
    generator = train_wgan(data, epochs_train)
    
    #create images of random noise
    file_name = 'generated_mask_' + str(len(file_names))
    save_path=path_mask_real.split('/')[0] + '/' + file_name
    os.mkdir(save_path)
    num_gen_data=num_data_gen
    device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")

    i=0
    img=[]
    while i < num_gen_data:
        img=[]
        fixed_noise = torch.randn(500, 100, 1, 1, device=device)
        img = generator(fixed_noise)
        i += 500
        
        if (i>num_gen_data):
            fixed_noise = torch.randn(num_gen_data - (i-500), 100, 1, 1, device=device)
            img = generator(fixed_noise)
            50
        for j in range(0, len(img)):
            torchvision.utils.save_image(img[j][0], str(save_path)+'/mask_gen_' + str(i-500+j) + ".png")