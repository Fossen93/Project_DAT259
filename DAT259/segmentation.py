from fastai.vision import *
import torchvision
import DAT259.setup as setup

def merge_df(path, img_folder, seg_folder, merge_with, number_of_gen_data):
    org_df = create_df_from_selected_files(merge_with)
    gen_df = create_seg_df(path, img_folder, seg_folder, number_of_gen_data)
    return org_df.append(gen_df)


def create_df_from_selected_files(df_used_seg):
    path = 'data/'
    img_folder = 'ISIC2018_Task1-2_Training_Input/'
    mask_folder = 'ISIC2018_Task1_Training_GroundTruth/'
    img = []
    mask = []

    for i, j in df_used_seg. iterrows():
        img_name = path + img_folder + j['Image']
        seg_name = path + mask_folder + j['Mask']
        img.append(img_name)
        mask.append(seg_name)
    """
    for i in range(len(df_used_seg)):
        x = str(df_used_seg.iloc[i,0])
        img_name = path +img_folder + (x.split('_')[0]+ '_' + x.split('_')[1] + '.jpg')
        seg_name = path + mask_folder + x
        img.append(img_name)
        mask.append(seg_name)
    """    
    return pd.DataFrame({"Image":img, "Mask":mask})


def create_seg_df(path, img_folder, seg_folder, num_of_data):
    
    img = []
    mask = []
    
    img_path = os.listdir(img_folder)
    for j in range(num_of_data-1):
        
        if(img_path[j].endswith('.png') | img_path[j].endswith('.jpg')): 
                
            img_name = str(img_folder)+ '/' + img_path[j]
            seg_name = str(seg_folder) + '/' + find_mask(img_path[j])
            img.append(img_name)
            mask.append(seg_name)
            #print(seg_name)
    return pd.DataFrame({"Image":img, "Mask":mask})


def find_mask(i):
    if i.startswith('ISIC'):
        new_name = i.split('.')[0] + '_segmentation.png'
    else:
        new_name = 'mask_' + i.split('_')[1] + '_' + i.split('_')[2]
    return new_name


def data_mix(data_path, df_inn, bs=8):
    
    path = Path(data_path)
    #print (path)
    # Create a new column for is_valid
    #df['is_valid'] = [True]*(df.shape[0]//2) + [False]*(df.shape[0]//2)

    # Randomly shuffle dataframe
    #df = df_inn.reindex(np.random.permutation(df_inn.index))
    df = df_inn
    
    #get_y_fn = lambda x: path_lbl/f'{x.stem}_segmentation.png'
    
    
    #decides the batch size based on GPU memory availabel
    #free = gpu_mem_get_free_no_cache()
    # the max size of bs depends on the available GPU RAM
    #if free > 8200: bs=8
    #else:           bs=4
    #print(f"using bs={bs}, have {free}MB of GPU RAM free")
    

    class SegLabelListCustom(SegmentationLabelList):
        def open(self, fn): return open_mask(fn, div=True)

    class SegItemListCustom(SegmentationItemList):
        _label_cls = SegLabelListCustom
    
    codes = ['Black', 'White']
    
    
    #src = (SegItemListCustom.from_folder(path).split_by_folder(image_folder, 'Test_img' ).label_from_func(get_y_fn, classes=codes))
    src = (SegItemListCustom.from_df(df, "")
            .split_by_rand_pct(0.1, seed=42)
            .label_from_df('Mask', classes=codes))
    
    
    data = (src.transform(tfms=get_transforms(), size=[128,128], tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

    return data

def create_learner(data):
    
    #types of algorithm to show how good the model is
    metrics=[accuracy_thresh]

    #weight decay
    wd=1e-2
    
    return unet_learner(data, models.resnet34, metrics=metrics, wd=wd)


def seg_model(learn, model_name):
    
    lr = 1e-3
    learn.fit_one_cycle(1, lr, pct_start=0.9)
    learn.unfreeze()
    
    
    lrs = slice(lr/400,lr/4)
    
    learn.fit_one_cycle(1, lrs, pct_start=0.8)
    
    learn.save(model_name)


def predict_on_test_data(learner, save_at):
    
    path_test_img = 'data/128x128/test_img'
    if (os.path.isdir(path_test_img)==False): os.mkdir(path_test_img)
    #hent test dataen
    test_img_list = os.listdir(path_test_img)
    for i in range(len(test_img_list)):
        
        img_name = test_img_list[i]
        full_img_path = path_test_img + '/' + img_name
        
        pred_name = img_name.split('.')[0] + '_prediction.png'
        full_pred_path = save_at + pred_name
        
        # prediker på bilde mappen
        img = open_image(full_img_path)
        pred_img = learner.predict(img)
        # lagre bildene i riktig mappe for predictions
        torchvision.utils.save_image(pred_img[1], full_pred_path)

def resize_img_folder(from_folder, to_folder, size, folder_name):

    setup.create_folder(to_folder)
    for i in folder_name:
        img_path = str(from_folder)+ '/' +i
        img = open_image(img_path).apply_tfms(tfms=None,size=size)
        save_as = str(to_folder)+ '/' + i
        img.save(save_as)


def dice_score (path_pred, path_targ):
    pred_names = os.listdir(path_pred)
    pred_names.sort()
    targ_names = os.listdir(path_targ)
    targ_names.sort()
    
    dice_results = []
    for i in range (len (pred_names)):    
        pred = open_mask(path_pred + '/' + pred_names[i], div = True)
        targ = open_mask(path_targ + '/' + targ_names[i], div = True)
        
        dice_coeff = dice(np.array(targ.data[0]), np.array(pred.data[0]))
        dice_results.append(dice_coeff)
    return sum(dice_results)/len(pred_names)


def dice(gt, img):
    """
    Input:
        gt: ground truth image data (as array)
        img: image to compare to gt (as array)
    Output: the Dice coefficient 
            https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient#Formula
    """
    # Find where nonzero:
    gt, img = [x > 0 for x in (gt, img)]
    
    # Numerator: 2 |gt intersect img|
    numerator = 2 * np.sum(gt & img)
    
    # Denominator: |gt| + |img|
    denominator = gt.sum() + img.sum()
    
    return numerator / denominator
