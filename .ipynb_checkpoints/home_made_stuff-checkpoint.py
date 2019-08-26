import matplotlib.pyplot as plt
from os import listdir
from PIL import Image as PImage
import numpy as np
import cv2
import os
import torchvision
from torchvision.utils import save_image
import torch
import pandas as pd


def get_img(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

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

def not_blank(img):
    is_it_blank = False
    if(np.unique(img).size > 1):
        is_it_blank = True
    return is_it_blank


def delete_img(full_img_path):
    os.remove(full_img_path)


def save_img(full_img_path, new_img):
    torchvision.utils.save_image(torch.from_numpy(new_img), full_img_path)

def name_fixer_3000(path, new_list, old_list):
    
    for i in range(len(new_list)):
        old_path = path + old_list[i]
        new_path = path + new_list[i]
        os.rename(new_path, old_path)  

# metode for å rydde opp i de genererte bildene, bare gi den sien til mappen med bildene som en streng, husk å avslutte strengen med '/'
def image_cleaner_3000(path):
    #lag imagelist
    imagelist = listdir(path)
    
    #iterer igjennom listen
    for i in range(len(imagelist)):
        full_img_path = path+imagelist[i]
        
        img = get_img(full_img_path)
        delete_img(full_img_path)
        
        if(not_blank(img)):
            new_img = undesired_objects(img)
            save_img(full_img_path, new_img)
            
    new_imagelist = listdir(path)
    name_fixer_3000(path,new_imagelist, imagelist)
    
    

# en metode for å generere en daframe, med en colononne som heter names
def create_df(path, folders):
    temp2 = []
    for i in range(len(folders)):
        kano = os.listdir(path+folders[i] + '/')
        for j in range(len(kano)):
            temp = folders[i]+ '/' + kano[j]
            temp2.append(temp)
    return pd.DataFrame({"names":temp2})






