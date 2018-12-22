import unet
import progressbar
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import sys, os, pickle, PIL, torch
from scipy import ndimage
import pandas as pd
import numpy as np, cv2 as cv
from sklearn.model_selection import train_test_split
from unet import detection_train, unet_train, ships_mask_dataset, test_image_dataset, mask_tr_no_aug

def split_mask(mask):
    threshold_obj = 60 #ignor predictions composed of "threshold_obj" pixels or less
    s=ndimage.generate_binary_structure(2,2)
    labled,n_objs = ndimage.label(mask,structure=s)
    result = []
    for i in range(n_objs):
        obj = (labled == i + 1).astype(int)
        if(obj.sum() > threshold_obj): result.append(obj)
    return result

def decode_mask(mask, shape=(768, 768)):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

#makes prediction for a pile of images
def non_empty(rn,dataloader):
    ans=[]
    rn.to(device)
    rn.eval()
    for sample in progressbar.progressbar(dataloader):
        name,img=sample['id'], sample['image']
        N, _, _, _=img.shape
        img=img.to(device)
        detect=rn(img)['class']
        detect=detect.to('cpu')
        detect=np.squeeze(torch.max(detect,1)[1].numpy())
        assert detect.shape==(N,), 'detect shape is wrong'
        indices=(np.nonzero(detect)[0])
        ans.extend([name[i] for i in indices])
    return ans
        
def infere(rn,un,dataloader):
    df=pd.DataFrame()
    rn.to(device)
    rn.eval()
    un.to(device)
    un.eval()
    assert(rn.training==False and un.training==False)
    for sample in progressbar.progressbar(dataloader):
        names,img=sample['id'], sample['image']
        img=img.to(device)
        N,_,_,_=img.shape
        mask=un(img)
        mask=mask.exp()>0.5
        mask=mask.to('cpu').numpy()
        for i in range(N):
            bigmask=cv.resize(mask[i,:,:],(ORIG_IMG_SIZE,ORIG_IMG_SIZE))
            components=split_mask(bigmask)
            if components==[]:
                cur=pd.DataFrame({'ImageId':names[i],'EncodedPixels':[np.NaN]})
                df=pd.concat([df,cur])
            else:
                for component in components:
                    cur=pd.DataFrame({'ImageId':names[i],'EncodedPixels':[decode_mask(component)]})
                    df=pd.concat([df,cur])
    return df

#print(os.getcwd())
TRAIN = '/floyd/input/ships/train/'
TEST = '/floyd/input/ships/test/'
LOGS= '/floyd/home/logs/'
LABELS='/floyd/input/ships/train_ship_segmentations_v2.csv'
ORIG_IMG_SIZE=768
IMG_SIZE=224
BATCH_SIZE=128
LEARNING_RATE=0.001
device='cuda'
exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted image
all_names = [f for f in os.listdir(TRAIN)]
test_names = [f for f in os.listdir(TEST)]
#test_names=test_names[0:100]
excluded_test_names=[f for f in test_names if f in exclude_list]

sample_pd=pd.read_csv('/floyd/input/ships/sample_submission_v2.csv')
sample_test_names=list(sample_pd['ImageId'])

for el in exclude_list:
    if(el in all_names): all_names.remove(el)
    if(el in test_names): test_names.remove(el)

        
test_images=test_image_dataset(test_names)
test_images_loader=DataLoader(test_images,batch_size=128)



g=detection_train(0)
rn,valloss,accuracy=g['model'],g['val_loss'],g['val_accuracy']
print('Loaded ResNet')
non_empty_test_names=non_empty(rn,test_images_loader)
print('nonempty_names # ',len(non_empty_test_names))

non_empty_test_images=test_image_dataset(non_empty_test_names)
non_empty_test_images_loader=DataLoader(non_empty_test_images,batch_size=10)

g=unet_train(rn,0)
un,valloss,iou=g['model'],g['val_loss'],g['val_iou']
nr=25
#src='test'
#img, name=test_images[nr]['image'], test_images[nr]['id']
src='train'
#img, name=mask_val_no_aug_set[nr]['image'], mask_val_no_aug_set[nr]['id']
non_empty_ans=infere(rn,un,non_empty_test_images_loader)
print('total names in non_empty ans',len(list(set(list(non_empty_ans['ImageId'])))))

empty_names=[name for name in test_names if name not in non_empty_test_names]
print('empty_names #',len(empty_names))
#nado 15606 ok


empty_ans=pd.DataFrame({'ImageId':empty_names,'EncodedPixels':[np.NaN for name in empty_names]})
print('empty ans shape',empty_ans.shape)
print('total names in empty ans',len(list(set(list(empty_ans['ImageId'])))))

ans=pd.concat([empty_ans,non_empty_ans])
print('ans shape',ans.shape)
ans.sort_index(inplace=True)
print('sorted ans shape',ans.shape)
#print(ans)
print('total names',len(list(set(list(ans['ImageId'])))))

ans.to_csv('output.csv',index=False)


sys.exit()

print('hule')
src='test'
name='462549a01.jpg'
img=cv.imread(os.path.join(TEST,name))
img = cv.resize(img,(IMG_SIZE,IMG_SIZE))
img=transforms.ToTensor()(img)
img=img.view(1,3,224,224)
img=img.to(device)
rn.to(device)
print('detection prediction')
print(rn(img)['class'].exp())

un.to(device)
un.eval()
ans=un(img)
ans=ans.exp()>0.5
ans=ans.to('cpu')
ans=ans.numpy()
ans=cv.resize(ans,(ORIG_IMG_SIZE,ORIG_IMG_SIZE))
from PIL import Image
strmask=decode_mask(ans)
print('predicted mask')
print(strmask)
print('after splitting')
for x in split_mask(ans):
    print(decode_mask(x))
ans = Image.fromarray(np.uint8((ans)*255))
ans.save('mask.png')
os.system('cp /floyd/input/ships/'+src+'/'+name+' /floyd/home/gauno.jpg')

