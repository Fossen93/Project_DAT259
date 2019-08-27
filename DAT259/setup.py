import matplotlib.pyplot as plt
from os import listdir
from PIL import Image as PImage
import numpy as np
import os
import torchvision
from torchvision.utils import save_image
import torch
import pandas as pd

def create_testset(path_img, path_mask, num_data):
    test_data = choose_data(path_mask, num_data)
    save_path_img = 'data/test_img'
    save_path_mask = 'data/test_mask'
    
    if (os.path.isdir(save_path_img)==False): os.mkdir(save_path_img)
    if (os.path.isdir(save_path_mask)==False): os.mkdir(save_path_mask)

    for i in test_data['Filenames']:
        os.rename(path_mask + '/' + i, save_path_mask + '/' + i)
        img_name = i.split('_')[0] + '_' + i.split('_')[1] + '.jpg'
        os.rename(path_img + '/' +img_name, save_path_img + '/' + img_name)
        
def choose_data(path, num_data=None):
    #get all the filenames
    file_names = os.listdir(path)
    index = []
    
    #cleans the filenames
    for i in range (len(file_names)):
        if (file_names[i].startswith('ISIC')==False):
            index.append(i)            

    file_names = np.delete(file_names, index)        
    
    np.random.shuffle(file_names)
    if(num_data==None):
        d = {'Filenames':file_names[:]}
        df = pd.DataFrame(data = d) 
    else:
        d = {'Filenames':file_names[:num_data]}
        df = pd.DataFrame(data = d)#file_names[: num_data], columns=['Filenames'])
    
    if (os.path.isdir('data/csv')==False): os.mkdir('data/csv')
    df.to_csv('data/csv/csv_'+ str(num_data))
    return df

def get_data(path):
    
    #df = pd.DataFrame()
    #df = df.from_csv(path)
    df = pd.read_csv(path, index_col = 0)
    return df