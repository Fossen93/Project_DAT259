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
import matplotlib.pyplot as plt
import cv2


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
            
            
def image_cleaner(path):
    #lag imagelist
    imagelist = listdir(path)
    #iterer igjennom listen
    for i in range(len(imagelist)):
        full_img_path = path+ '/' + imagelist[i]
        
        img = get_img(full_img_path)
        delete_img(full_img_path)
        
        if(not_blank(img)):
            new_img = undesired_objects(img)
            save_img(full_img_path, new_img)
            
    new_imagelist = listdir(path)
    name_fixer(path,new_imagelist, imagelist)


def get_img(img_path):
    img = cv2.imread(img_path)
    cv2.imshow('image', img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


def delete_img(full_img_path):
    os.remove(full_img_path)


def undesired_objects(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return img2


def save_img(full_img_path, new_img):
    torchvision.utils.save_image(torch.from_numpy(new_img), full_img_path)


def name_fixer(path, new_list, old_list):
    
    for i in range(len(new_list)):
        old_path = path + '/' + old_list[i]
        new_path = path + '/' + new_list[i]
        os.rename(new_path, old_path)  

def not_blank(img):
    is_it_blank = False
    if(np.unique(img).size > 1):
        is_it_blank = True
    return is_it_blank