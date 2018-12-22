import torch
import sys
import math
import cv2
import pickle
import random
import progressbar
import torch.nn.functional as F
import numpy as np
import cv2 as cv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import OrderedDict
from skimage import io, transform
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN = './data/train/' # train folder
TEST = './data/test/' # test folder
LOGS= './logs/' # folder to ouput trianing logs
LABELS='./data/train_ship_segmentations_v2.csv' # path to label data
ORIG_IMG_SIZE=768
IMG_SIZE=224
BATCH_SIZE=128
LEARNING_RATE=0.001
device='cuda'
exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted images
all_names = [f for f in os.listdir(TRAIN)]
test_names = [f for f in os.listdir(TEST)]
for el in exclude_list:
    if(el in all_names): all_names.remove(el)
    if(el in test_names): test_names.remove(el)

train_names, val_names = train_test_split(all_names, test_size=0.05, random_state=42)
segmentation_df = pd.read_csv(LABELS).set_index('ImageId')

def get_mask(img_id, df):
    '''
    Args:
        img_id: image id
        df: dataframe with mask in run-length encoding
    Returns:
        img: binary mask image with
    '''
    shape = (ORIG_IMG_SIZE,ORIG_IMG_SIZE)
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    masks = df.loc[img_id]['EncodedPixels']
    if(type(masks) == float): return img.reshape(shape)
    if(type(masks) == str): masks = [masks]
    for mask in masks:
        s = mask.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1
    return img.reshape(shape).T

# Dataset containing id, image and mask, augmented by applying transform
class ships_mask_dataset(Dataset):
    def __init__(self, names,transform=None):
        self.names=names
        self.transform=transform
    def __getitem__(self,id):
        name=self.names[id]
        image=cv.imread(os.path.join(TRAIN,name))
        mask=get_mask(name,segmentation_df)
        if self.transform:
            transformed=self.transform(image,mask)
            image, mask=transformed['image'], transformed['mask'].float()
        return {'id':name,'image':image,'mask':mask}
    def __len__(self):
        return len(self.names)

# Dataset containing id and image only
class ships_image_dataset(Dataset):
    def __init__(self, names):
        self.names=names
    def __getitem__(self,id):
        name=self.names[id]
        img=io.imread(os.path.join(TRAIN,name))
        return{'id':name,'image':img}
    def __len__(self):
        return len(self.names)

# Dataset for ship detection, contains id, image and class 0=no ship, 1=ship
class ships_detect_dataset(Dataset):
    def __init__(self, names,transform=None):
        self.names=names
        self.transform=transform
    def __getitem__(self,id):
        name=self.names[id]
        img=cv.imread(os.path.join(TRAIN,name))
        if type(segmentation_df.loc[name]['EncodedPixels'])==float:
            cl=0
        else:
            cl=1
        if self.transform:
            img=self.transform(img)
            cltens=torch.tensor(cl)
        return{'id':name,'image':img,'class':cltens}
    def __len__(self):
        return len(self.names)

# Transformation that transforms image and mask simultaneously
class transform(object):
    def __call__(self,image,mask):
        image = cv.resize(image,(IMG_SIZE,IMG_SIZE))
        mask = cv.resize(mask,(IMG_SIZE,IMG_SIZE))
        r=random.choice([1,0,-1])
        image=cv.flip(image,r)
        mask=cv.flip(mask,r)
        image=transforms.ToPILImage()(image)
        image=transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)(image)
        image=transforms.ToTensor()(image)
        mask=torch.from_numpy(mask)
        return {'image':image,'mask':mask}

# Transformation that transforms image only
class transform_img(object):
    def __call__(self,image):
        image = cv.resize(image,(IMG_SIZE,IMG_SIZE))
        r=random.choice([1,0,-1])
        image=cv.flip(image,r)
        image=transforms.ToPILImage()(image)
        image=transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)(image)
        image=transforms.ToTensor()(image)
        return image

# Construct train and validation set for ship detection
tr=transform_img()
detect_train_set=ships_detect_dataset(train_names,tr)
detect_val_set=ships_detect_dataset(val_names,tr)

# Construct train and validation set for mask prediction
mask_tr=transform()
mask_train_set=ships_mask_dataset(nonempty_train_names,mask_tr)
mask_val_set=ships_mask_dataset(nonempty_val_names,mask_tr)

resnet=models.resnet34(pretrained=True)

def resnet_feature_dim(size):
    assert size>=224,'image size must be >=224'
    x=torch.from_numpy(np.zeros((1,3,size,size))).float()
    x=resnet.conv1(x)
    x=resnet.bn1(x)
    x=resnet.relu(x)
    x=resnet.maxpool(x)
    x=resnet.layer1(x)
    x=resnet.layer2(x)
    x=resnet.layer3(x)
    x=resnet.layer4(x)
    x=resnet.avgpool(x)
    x=x.view(x.size(0),-1)
    return x.shape[1]
resnet_out_features=resnet_feature_dim(IMG_SIZE)

#resnet_out_features=165888
class myresnet(nn.Module):
    def __init__(self,hidden_layers):
        super(myresnet, self).__init__()
        self.backbone=resnet
        self.classifier=nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(resnet_out_features, hidden_layers)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_layers, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    def forward(self,x):
        x=self.backbone.conv1(x)
        x=self.backbone.bn1(x)
        x=self.backbone.relu(x)
        l0=self.backbone.maxpool(x)
        l1=self.backbone.layer1(l0)
        l2=self.backbone.layer2(l1)
        l3=self.backbone.layer3(l2)
        l4=self.backbone.layer4(l3)
        out=self.backbone.avgpool(l4)
        out=out.view(out.size(0),-1)
        out=self.classifier(out)
        return {'layer0':l0,'layer1':l1,'layer2':l2,'layer3':l3,'layer4':l4,'class':out}

def detection_train(epochs):
    rn=myresnet(512)
    wght=torch.Tensor([positive_ratio,negative_ratio])
    wght=wght.to(device)
    dloss=nn.NLLLoss(weight=wght)

    lrs=[rn.backbone.conv1,rn.backbone.bn1,rn.backbone.maxpool,rn.backbone.layer1,rn.backbone.layer2,rn.backbone.layer3]
    for l in lrs:
        for x in l.parameters():
            x.requires_grad=False
    active_parameters=[par for par in rn.parameters() if par.requires_grad==True]
    print('training active pars:',len(active_parameters))
    optimizer=optim.Adam(active_parameters,lr=LEARNING_RATE)

    train_loader=DataLoader(detect_train_set,batch_size=BATCH_SIZE)
    val_loader=DataLoader(detect_val_set,batch_size=BATCH_SIZE)

    #print(type(rn))
    rn.to(device)
    current_epoch=0
    best_val_loss=None
    train_loss=[]
    val_loss=[]
    val_accuracy=[]
    MODEL_PATH=os.path.join(LOGS,'best.model.pth')
    if os.path.isfile(MODEL_PATH):
        checkpoint=torch.load(MODEL_PATH)
        current_epoch=checkpoint['epoch']
        best_val_loss=checkpoint['loss']
        train_loss=checkpoint['train_loss']
        val_loss=checkpoint['val_loss']
        if 'val_accuracy' in checkpoint.keys():
            val_accuracy=checkpoint['val_accuracy']
        print('loading model from',MODEL_PATH)
        rn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    print('start training')
    for epoch in range(current_epoch,epochs):
        print('epoch',epoch)
        running_loss=0
        num_batches=0
        rn.train()
        trlen=len(train_loader)
        for sample in train_loader:
            image,cls=sample['image'],sample['class']
            image,cls=image.to(device),cls.to(device)
            optimizer.zero_grad()
            predict=rn(image)['class']
            loss=dloss(predict,cls)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            #if math.isnan(loss.item()):
            #    print(sample['id'])
            #    break
            print(str(num_batches)+'/'+str(trlen)+' batch loss',loss.item())
            num_batches+=1
        print('epoch ',epoch,', train loss:',running_loss/num_batches)
        train_loss.append(running_loss/num_batches)
        running_loss=0
        num_batches=0
        num_examples=0
        TP=0
        PREDICTED=0
        REALPOSITIVE=0
        CORRECT=0
        rn.eval()
        for sample in val_loader:
            image,cls=sample['image'],sample['class']
            image,cls=image.to(device),cls.to(device)
            predict=rn(image)['class']
            loss=dloss(predict,cls)
            running_loss+=loss.item()
            num_batches+=1
            num_examples+=cls.size(0)
            predict=torch.max(predict,1)[1]
            CORRECT+=(predict==cls).sum().item()
            TP+=(predict*cls).sum().item()
            PREDICTED+=predict.sum().item()
            REALPOSITIVE+=cls.sum().item()
        valloss=running_loss/num_batches
        try:
            precision=TP/(PREDICTED)
            recall=TP/(REALPOSITIVE)
        except:
            precision=1
            recall=1
        print('num_examples',num_examples)
        print('epoch ',epoch,', val loss:',valloss, 'val accuracy', CORRECT/num_examples, 'val f1 score',2*precision*recall/(precision+recall))

        val_loss.append(valloss)
        val_accuracy.append(CORRECT/num_examples)
        if not best_val_loss:
            best_val_loss=valloss
            torch.save({'loss': valloss,'epoch': epoch,'model_state_dict': rn.state_dict(),'optimizer_state_dict': optimizer.state_dict(),\
            'train_loss':train_loss,'val_loss':val_loss,'val_accuracy':val_accuracy},MODEL_PATH)
        elif valloss<best_val_loss:
            best_val_loss=valloss
            torch.save({'loss': valloss,'epoch': epoch,'model_state_dict': rn.state_dict(),'optimizer_state_dict': optimizer.state_dict(),\
            'train_loss':train_loss,'val_loss':val_loss,'val_accuracy':val_accuracy},MODEL_PATH)
    return {'model':rn,'val_loss':val_loss,'val_accuracy':val_accuracy}




class UNet(nn.Module):
    def __init__(self,rene):
        super(UNet,self).__init__()
        self.rene=rene
        self.ct2d=nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.c2d=nn.Conv2d(512,256,kernel_size=3,padding=1,stride=1)
        self.bn1=nn.BatchNorm2d(256)
        self.relu=nn.ReLU()
        self.ct2d2=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.c2d2=nn.Conv2d(256,128,kernel_size=3,padding=1,stride=1)
        self.bn2=nn.BatchNorm2d(128)
        self.ct2d3=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.c2d3=nn.Conv2d(128,64,kernel_size=3,padding=1,stride=1)
        self.bn3=nn.BatchNorm2d(64)
        self.ct2d4=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.c2d4=nn.Conv2d(64,32,kernel_size=3,padding=1,stride=1)
        self.bn4=nn.BatchNorm2d(32)
        self.ct2d5=nn.ConvTranspose2d(32,16,kernel_size=2,stride=2)
        self.c2d5=nn.Conv2d(16,1,kernel_size=1,stride=1)

    def forward(self,x):
        g=self.rene(x)
        l0,l1,l2,l3,l4,out=g['layer0'],g['layer1'],g['layer2'],g['layer3'],g['layer4'],g['class']
        #l4=l4.to('cpu')
        up=self.ct2d(l4)#increase size from IMG_SIZE/32 to IMG_SIZE/16
        up=torch.cat([l3,up],dim=1) # size Nx512xIMG/SIZE/16**2
        up=self.c2d(up) # size Nx256xIMG_SIZE/16**2
        up=self.bn1(up)
        up=self.relu(up)
        up=self.ct2d2(up) # size Nx128xIMG_SIZE/8**2
        up=torch.cat([l2,up],dim=1) # size Nx256xIMG_SIZE/8**2
        up=self.c2d2(up) # size Nx128xIMG_SIZE/8**2
        up=self.bn2(up)
        up=self.relu(up)
        up=self.ct2d3(up) # size Nx64xIMG_SIZE/4**2
        up=torch.cat([l1,up],dim=1) # size Nx128xIMG_SIZE/4**2
        up=self.c2d3(up) # size Nx64xIMG_SIZE/4**2
        up=self.bn3(up)
        up=self.relu(up)
        up=torch.cat([l0,up],dim=1) # size Nx128xIMG_SIZE/4**2
        up=self.ct2d4(up) # size Nx64xIMG_SIZE/2**2
        up=self.c2d4(up) # size Nx16xIMG_SIZE/2**2
        up=self.bn4(up)
        up=self.relu(up)
        up=self.ct2d5(up) # size Nx8xIMG_SIZE**2
        up=self.c2d5(up)
        up=torch.squeeze(up)
        up=nn.LogSigmoid()(up)
        return up





#take gamma=2
#training UNet:
#Loss function
#class FocalLoss(nn.Module):
#    def __init__(self, gamma):
#        super().__init__()
#        self.gamma = gamma
#    def forward(self,input,target):
# Input=log(p), target=y=1 or 0
# pt=p if y=1 of (1-p) if y=0
# FL=-(1-pt)**g*log(pt) inverse=log(1-p)
#        inverse=(1-input.exp()).log()
#        FL=(inverse*self.gamma).exp()*input*target+#(input*self.gamma).exp()*inverse*(1-target)
#        return FL.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input-input*target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input*(target*2.0 - 1.0))
        loss = (invprobs*self.gamma).exp()*loss

        return loss.mean()

def DiceLoss(input,target):
    eps=1
    input=input.exp()
    iflat = input.view(-1)
    tflat = target.view(-1)
    dl=1-(iflat*tflat.sum()+eps)/(iflat.sum()+tflat.sum()+eps)-(((1-iflat)*(1-tflat).sum()+eps)/((1-iflat).sum()+(1-tflat).sum()+eps))
    return dl

class Loss(nn.Module):
    def __init__(self,alpha,gamma):
        super().__init__()
        self.focal=FocalLoss(gamma)
        self.alpha=alpha
    def forward(self,input,target):
        loss=self.alpha*self.focal(input,target)-torch.log(DiceLoss(input,target))
        return loss.mean()


train_mask_loader=DataLoader(mask_train_set,batch_size=BATCH_SIZE)
val_mask_loader=DataLoader(mask_val_set,batch_size=BATCH_SIZE)


def unet_train(rn,epochs):
    un=UNet(rn)
    un=un.to(device)
    maskloss=Loss(alpha=1,gamma=2)

    active_parameters=[x for x in un.parameters() if x.requires_grad==True]
    optimizer=optim.Adam(active_parameters,lr=LEARNING_RATE)
    print('training ',len(active_parameters),' active parameters')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=False, threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    current_epoch=0
    best_val_loss=None
    train_loss=[]
    val_loss=[]
    MODEL_PATH=os.path.join(LOGS,'best_unet_model.pth')
    if os.path.isfile(MODEL_PATH):
        checkpoint=torch.load(MODEL_PATH)
        current_epoch=checkpoint['epoch']
        best_val_loss=checkpoint['loss']
        train_loss=checkpoint['train_loss']
        val_loss=checkpoint['val_loss']
        print('loading model from',MODEL_PATH)
        un.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    print('start training UNet')
    for epoch in range(current_epoch,epochs):
        print('epoch',epoch)
        running_loss=0
        num_batches=0
        un.train()
        trlen=len(train_mask_loader)
        for sample in train_mask_loader:
            image,mask=sample['image'],sample['mask']
            image,mask=image.to(device),mask.to(device)
            optimizer.zero_grad()
            predict=un(image)
            loss=maskloss(predict,mask)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            #if math.isnan(loss.item()):
            #    print(sample['id'])
            #    break
            print(str(num_batches)+'/'+str(trlen)+' batch loss',loss.item())
            num_batches+=1
        print('epoch ',epoch,', train loss:',running_loss/num_batches)
        train_loss.append(running_loss/num_batches)
        running_loss=0
        num_batches=0
        un.eval()
        for sample in val_mask_loader:
            image,mask=sample['image'],sample['mask']
            image,mask=image.to(device),mask.to(device)
            predict=un(image)
            mask=torch.squeeze(mask)
            loss=maskloss(predict,mask)
            running_loss+=loss.item()
            num_batches+=1

        valloss=running_loss/num_batches
        scheduler.step(valloss)
        print('epoch ',epoch,', val loss:',valloss)

        val_loss.append(valloss)
        if not best_val_loss:
            best_val_loss=valloss
            torch.save({'loss': valloss,'epoch': epoch,'model_state_dict': un.state_dict(),'optimizer_state_dict': optimizer.state_dict(),\
            'train_loss':train_loss,'val_loss':val_loss},MODEL_PATH)
        elif valloss<best_val_loss:
            best_val_loss=valloss
            torch.save({'loss': valloss,'epoch': epoch,'model_state_dict': un.state_dict(),'optimizer_state_dict': optimizer.state_dict(),\
            'train_loss':train_loss,'val_loss':val_loss},MODEL_PATH)
    return {'model':un, 'val_loss':val_loss}


g=detection_train(0)
rn,valloss=g['model'],g['val_loss']
print('detection validation loss',valloss)

#Freeze the resnet parameters
for param in rn.parameters():
    rn.requires_grad=False

g=unet_train(rn,50)
un,valloss=g['model'],g['val_loss']
img=mask_train_set[0]['image']
img=img.view(1,3,224,224)
img=img.to(device)
mask=un(img)
mask=mask.exp()>0.5
mask=mask.to('cpu')
mask=mask.numpy()
mask=np.squeeze(mask)
mask=mask*244
print(mask)
mask=Image.fromarray(mask)
mask.save('mask.png')
