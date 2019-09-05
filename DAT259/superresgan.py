from fastai.vision import *
from fastai.vision.gan import *
import torchvision
#from torchvision.utils import save_image
from torchvision.models import vgg16_bn
from fastai.callbacks import *
import DAT259.setup as setup
 
base_loss = F.l1_loss

#Pretrain generator ------------------

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()



def train_generator(file_names, path_mask_real, path_img_real, save_path, bs=32, size=128, num_epochs = 10):
    file_names = file_names.drop(['Image'], axis = 1)
    data = get_data_superres(file_names, path_mask_real, path_img_real, bs, size)
    
    learn = create_generator(data)
    
    gc.collect()
    
    learn.fit_one_cycle(num_epochs, slice(2e-3), pct_start=0.9)
    
    learn.unfreeze()
    learn.fit_one_cycle(num_epochs, slice(1e-5,2e-3), pct_start=0.9)
    learn.save('generator_' + str(len(file_names)))
    learn.show_results(rows=10, imgsize=5)
    
    #folder_name = 'pretrained_masks' + str(len(file_names))
    #setup.create_folder(save)
    save_preds_l1(data.fix_dl, save_path, learn)
    
    return learn

def get_lbl(x, path):
    x = Path(x)
    x = x.stem
    x = str(x)
    s = x.split('_')[0]+ '_' + x.split('_')[1] + '.jpg'
    s = Path(s)
    return path/s

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


def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)

def save_preds_l1(dl, path, model):
    i=0
    names = dl.dataset.items
    #save_path=path
    setup.create_folder(path)
    for b in dl:
        preds = model.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            name = names[i].split('/')[-1]
            o.save(path/name)
            i += 1


# Pretrain critic ---------------------------

def train_critic(num_data, path_org, path_pretrained, num_epochs = 10, wd = 1e-3):
    file_names = {'Filenames': os.listdir(str(path_pretrained) + '/pretrained_images_' + str(num_data))}
    file_names = pd.DataFrame(data=file_names)
    
    labeled_data= label_data(file_names, path_org, path_pretrained, ['pretrained_images_' + str(num_data), "ISIC2018_Task1-2_Training_Input"])
    
    data_crit = get_crit_data(labeled_data)
    
    learn_critic = create_critic_learner(data_crit, accuracy_thresh_expand, wd)
    
    learn_critic.fit_one_cycle(num_epochs, 1e-3)
    
    learn_critic.save('critic_' + str(num_data))


def label_data(df, path_org, path_pretrained, classes):
    
    labeled_data = pd.DataFrame(columns=['Filenames', 'label'])
    for i in classes:
        for j in df['Filenames']:#range (len(df)):
            if i.split('_')[-1] == 'Input':
                name = j.split('_')[0] +'_'+j.split('_')[1]+'.jpg'
                labeled_data = labeled_data.append({'Filenames':str(path_org)+'/'+i+'/'+name, 'label':i}, ignore_index=True)
            else:
                labeled_data = labeled_data.append({'Filenames':str(path_pretrained)+'/'+i+'/'+j, 'label':i}, ignore_index=True)
            
        
    return labeled_data

def get_crit_data(df, bs=32, size=128):
    src = ImageList.from_df(path = Path(''), df=df).split_by_rand_pct(0.1, seed=42)
    ll = src.label_from_df('label')
    data = (ll.transform(get_transforms(), size=size)
           .databunch(bs=bs).normalize(imagenet_stats))
    data.c = 3
    return data

def create_critic_learner(data, metrics, wd = 1e-3, loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())):
    return Learner(data, gan_critic(), metrics=metrics, loss_func=loss_critic, wd=wd)


# Train GAN ----------------------------

def train_gan (file_names, path_mask_real, path_img_real, num_epochs=100, bs=32, size=128, wd=1e-3):
    file_names = file_names.drop(['Image'], axis = 1)
    file_names = file_names.rename(columns={"Mask": "Filenames"})
    
    learn_crit=None
    learn_gen=None
    gc.collect()
    
    labeled_data= label_data(file_names, Path('data'), Path('data'), ['ISIC2018_Task1_Training_GroundTruth', "ISIC2018_Task1-2_Training_Input"])

    data_crit = get_crit_data(labeled_data, bs=bs, size=size)
    learn_crit = create_critic_learner(data_crit, metrics=None).load('critic_' + str(len(file_names)))
    
    
    data_gen = get_data_superres(file_names,path_mask_real, path_img_real)
    learn_gen = create_generator(data_gen)
    
    learn_gen.load('generator_' + str(len(file_names)))
    
    switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
    learn = GANLearner.from_learners(learn_gen, learn_crit, weights_gen=(1.,50.), show_img=False, switcher=switcher,
                                 opt_func=partial(optim.Adam, betas=(0.,0.99)), wd=wd)
    learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))
    
    learn.fit(num_epochs, 1e-4)
    learn.show_results(rows=8)
    
    return learn_gen


def pred_gan (model, path_mask, path_gen, num_data):
    #os.mkdir(path_gen)
    
    for i in range(num_data):
        #try:
            img = open_image(str(path_mask) + '/mask_gen_' + str(i) + '.png')
            a, pred_img, b = model.predict(img)
            torchvision.utils.save_image(pred_img, str(path_gen) + '/img_gen_' + str(i) + '.png')
        #except:
         #   print ("exception")