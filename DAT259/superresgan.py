from fastai.vision import *

def train_generator(file_names, path_mask_real, path_img_real, bs=32, size=128):
    
    data = get_data_superres(file_names, path_mask_real, path_img_real, bs, size)
    
    learn = create_generator(data)
    
    gc.collect();
    
    learn.fit_one_cycle(10, slice(2e-3), pct_start=0.9)
    
    learn.unfreeze()
    learn.fit_one_cycle(10, slice(1e-5,2e-3), pct_start=0.9)
    learn.save('generator_' + str(len(file_names)))
    learn.show_results(rows=10, imgsize=5)
    
    folder_name = 'gen_l1_' + str(len(file_names))
    save_preds_l1(data.fix_dl, Path('data/' + folder_name), learn)
    
    return learn


def get_data_superres(df, path_mask_real, path_img_real, bs=32, size=128):
    src = ImageImageList.from_df(path=path_mask_real, df=df).split_by_rand_pct(0.1, seed=42)
    
    data = (src.label_from_func(lambda x: get_lbl(x, path_img_real))
           .transform(get_transforms(), size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    
    return data


def create_generator(data):
    
    arch = models.resnet34
    
    t = data.valid_ds[0][1].data
    t = torch.stack([t,t])
    
    base_loss = F.l1_loss

    vgg_m = vgg16_bn(True).features.cuda().eval()
    requires_grad(vgg_m, False)
    
    blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
    
    feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])
    
    wd = 1e-3
    learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,
                     blur=True, norm_type=NormType.Weight)
    return learn