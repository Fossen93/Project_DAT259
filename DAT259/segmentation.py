from fastai.vision import *
import torchvision

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
    
    for i in range(len(df_used_seg)):
        x = str(df_used_seg.iloc[i,0])
        img_name = img_folder + (x.split('_')[0]+ '_' + x.split('_')[1] + '.jpg')
        seg_name = path + mask_folder + x
        img.append(img_name)
        mask.append(seg_name)
    
    return pd.DataFrame({"img":img, "mask":mask})


def create_seg_df(path, img_folder, seg_folder, num_of_data):
    
    img = []
    mask = []
    
    img_path = os.listdir(path+img_folder + '/')
    for j in range(num_of_data-1):
        
        if(img_path[j].endswith('.png') | img_path[j].endswith('.jpg')): 
                
            img_name = img_folder+ '/' + img_path[j]
            seg_name = path + seg_folder + '/' + find_mask(img_path[j])
            img.append(img_name)
            mask.append(seg_name)
    
    return pd.DataFrame({"img":img, "mask":mask})


def find_mask(i):
    if i.startswith('ISIC'):
        new_name = i.split('.')[0] + '_segmentation.png'
    else:
        new_name = 'mask_' + i.split('_')[1] + '_' + i.split('_')[2]
    return new_name


def data_mix(data_path, df_inn, bs=8):
    
    path = Path(data_path)
    
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
    src = (SegItemListCustom.from_df(df, path)
            .split_by_rand_pct(0.1, seed=42)
            .label_from_df('mask', classes=codes))
    
    
    data = (src.transform(tfms=get_transforms(), size=[128,128], tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))
    
    return data

def create_learner(data):
    
    #types of algorithm to show how good the model is
    metrics=[accuracy_thresh, dice]

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
    if (os.path.isdir('data/128x128')==False): os.mkdir('data/128x128')
    path_test_img = 'data/128x128/test_img'
    if (os.path.isdir(path_test_img)==False): os.mkdir(path_test_img)
    #hent test dataen
    test_img_list = os.listdir(path_test_img)
    print(len(test_img_list))
    for i in range(len(test_img_list)):
        print ("hei")
        img_name = test_img_list[i]
        full_img_path = path_test_img + img_name
        
        pred_name = img_name.split('.')[0] + '_prediction.png'
        full_pred_path = save_at + pred_name
        
        # prediker på bilde mappen
        img = open_image(full_img_path)
        pred_img = learner.predict(img)
        # lagre bildene i riktig mappe for predictions
        torchvision.utils.save_image(pred_img[1], full_pred_path)