import unet
import progressbar
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import sys
import os
import pickle
import PIL
import torch
from PIL import Image
from scipy import ndimage
import pandas as pd
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split
from unet import detection_train, unet_train, ships_mask_dataset, test_image_dataset, mask_tr_no_aug


def split_mask(mask):
    threshold_obj = 60  # ignor predictions composed of "threshold_obj" pixels or less
    s = ndimage.generate_binary_structure(2, 2)
    labled, n_objs = ndimage.label(mask, structure=s)
    result = []
    for i in range(n_objs):
        obj = (labled == i + 1).astype(int)
        if(obj.sum() > threshold_obj):
            result.append(obj)
    return result


def decode_mask(mask, shape=(768, 768)):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def non_empty(rn, dataloader):
    '''predicts if image contains a ship'''
    ans = []
    rn.to(device)
    rn.eval()
    for sample in progressbar.progressbar(dataloader):
        name, img = sample['id'], sample['image']
        N, _, _, _ = img.shape
        img = img.to(device)
        detect = rn(img)['class']
        detect = detect.to('cpu')
        detect = np.squeeze(torch.max(detect, 1)[1].numpy())
        assert detect.shape == (N,), 'detect shape is wrong'
        indices = (np.nonzero(detect)[0])
        ans.extend([name[i] for i in indices])
    return ans


def infere(rn, un, dataloader):
    ''' predicts masks in rle encoding'''
    df = pd.DataFrame()
    rn.to(device)
    rn.eval()
    un.to(device)
    un.eval()
    assert(rn.training == False and un.training == False)
    for sample in progressbar.progressbar(dataloader):
        names, img = sample['id'], sample['image']
        img = img.to(device)
        N, _, _, _ = img.shape
        mask = un(img)
        mask = mask.exp() > 0.5
        mask = mask.to('cpu').numpy()
        for i in range(N):
            bigmask = cv.resize(mask[i, :, :], (ORIG_IMG_SIZE, ORIG_IMG_SIZE))
            components = split_mask(bigmask)
            if components == []:
                cur = pd.DataFrame(
                    {'ImageId': names[i], 'EncodedPixels': [np.NaN]})
                df = pd.concat([df, cur])
            else:
                for component in components:
                    cur = pd.DataFrame(
                        {'ImageId': names[i], 'EncodedPixels': [decode_mask(component)]})
                    df = pd.concat([df, cur])
    return df


TRAIN = '/floyd/input/ships/train/'
TEST = '/floyd/input/ships/test/'
LOGS = '/floyd/home/logs/'
LABELS = '/floyd/input/ships/train_ship_segmentations_v2.csv'
ORIG_IMG_SIZE = 768
IMG_SIZE = 224
BATCH_SIZE = 128
LEARNING_RATE = 0.001

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

exclude_list = ['6384c3e78.jpg', '13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']  # corrupted image
all_names = [f for f in os.listdir(TRAIN)]
test_names = [f for f in os.listdir(TEST)]
# test_names=test_names[0:100]
excluded_test_names = [f for f in test_names if f in exclude_list]

sample_pd = pd.read_csv('/floyd/input/ships/sample_submission_v2.csv')
sample_test_names = list(sample_pd['ImageId'])

for el in exclude_list:
    if el in all_names:
        all_names.remove(el)
    if el in test_names:
        test_names.remove(el)


test_images = test_image_dataset(test_names)
test_images_loader = DataLoader(test_images, batch_size=128)

g = detection_train(0)
rn, valloss, accuracy = g['model'], g['val_loss'], g['val_accuracy']
print('Loaded ResNet')


g = unet_train(rn, 0)
un, iou = g['model'], g['val_iou']

print('validation iou', iou)


src = 'test'
name1, name2 = '9bc4ea7cb.jpg', '009c7f8ec.jpg'
img1, img2 = cv.imread(os.path.join(TEST, name1)), cv.imread(
    os.path.join(TEST, name2))
img1, img2 = cv.resize(img1, (IMG_SIZE, IMG_SIZE)), cv.resize(
    img2, (IMG_SIZE, IMG_SIZE))
img1, img2 = transforms.ToTensor()(img1), transforms.ToTensor()(img2)
img1, img2 = img1.view(1, 3, 224, 224), img2.view(1, 3, 224, 224)
img1, img2 = img1.to(device), img2.to(device)
rn.to(device)

un.to(device)
un.eval()
ans1, ans2 = un(img1), un(img2)
ans1, ans2 = ans1.exp() > 0.5, ans2.exp() > 0.5
ans1, ans2 = ans1.to('cpu'), ans2.to('cpu')
ans1, ans2 = ans1.numpy(), ans2.numpy()
ans1 = cv.resize(ans1, (ORIG_IMG_SIZE, ORIG_IMG_SIZE))
ans2 = cv.resize(ans2, (ORIG_IMG_SIZE, ORIG_IMG_SIZE))

ans1 = Image.fromarray(np.uint8((ans1)*255))
ans2 = Image.fromarray(np.uint8((ans2)*255))
ans1.save('mask1.png')
ans2.save('mask2.png')
os.system('cp /floyd/input/ships/'+src+'/'+name1+' /floyd/home/testimage1.jpg')
os.system('cp /floyd/input/ships/'+src+'/'+name2+' /floyd/home/testimage2.jpg')
