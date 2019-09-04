import os
import numpy as np
import pandas as pd

def split_train_test (path_img, path_mask, num_test):
    test_data, train_data = choose_test(path_img, num_test)
    test_data = test_data.rename(columns={"Filenames": "Image"})
    test_data['Mask'] = test_data['Image']

    for i in range (len(test_data['Mask'])):
        name = test_data['Mask'][i].split('.')[0] + '_segmentation.png'
        test_data['Mask'][i] = name

    train_data = train_data.rename(columns={"Filenames": "Image"})
    train_data['Mask'] = train_data['Image']

    for i in range (len(train_data['Mask'])):
        name = train_data['Mask'][i].split('.')[0] + '_segmentation.png'
        train_data['Mask'][i] = name
        
    return train_data, test_data

def choose_test(path, num_data):
    file_names = os.listdir(path)
    index = []

    #cleans the filenames
    for i in range (len(file_names)):
        if (file_names[i].startswith('ISIC')==False):
            index.append(i)            

    file_names = np.delete(file_names, index)        
    
    np.random.shuffle(file_names)
    
    test_df = pd.DataFrame()
    test_df['Image'] = file_names[:num_data]
    rest_df = pd.DataFrame()
    rest_df['Image'] = file_names[num_data:]

    return test_df, rest_df

def create_folder (path):
    if (os.path.isdir(path)==False): os.mkdir(path)

