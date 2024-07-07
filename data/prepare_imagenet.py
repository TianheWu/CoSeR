import os
import blobfile as bf
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

target_hr_path = ''
os.makedirs(target_hr_path, exist_ok=True)

imagenet_path = ''
resolution = 512

class_path = os.listdir(imagenet_path)

for cp in tqdm(class_path):
    image_path = os.listdir(os.path.join(imagenet_path, cp))
    # print(f'Processing {i}')
    for ip in image_path:
        path = os.path.join(imagenet_path, cp, ip)
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        img_size = os.path.getsize(path)
        img_size = img_size/1024

        if pil_image.size[0]*pil_image.size[1] >= 384*384 and img_size>=50:
            while min(*pil_image.size) >= 2 * resolution:
                pil_image = pil_image.resize(
                    tuple(x // 2 for x in pil_image.size), resample=Image.BOX
                )
                
            scale = resolution / min(*pil_image.size)
            pil_image = pil_image.resize(
                tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
            )
            
            arr = np.array(pil_image.convert("RGB"))
            crop_y = (arr.shape[0] - resolution) // 2
            crop_x = (arr.shape[1] - resolution) // 2
            arr = arr[crop_y : crop_y + resolution, crop_x : crop_x + resolution]
            im = Image.fromarray(arr)
            im.save(os.path.join(target_hr_path, ip))

img_path = os.listdir(target_hr_path)
img_class = []

print(len(img_path))

for i in img_path:
    img_class.append(i.split('_')[0])

bb = pd.Series(img_class)
a = bb.value_counts()

print(len(a))

for i in a.index:
    a.loc[i] = 2

random.shuffle(img_path)

val_list = []
train_list = []

for i in img_path:
    class_name = i.split('_')[0]
    if a.loc[class_name] > 0:
        val_list.append(i)
        a.loc[class_name] = a.loc[class_name] - 1
    else:
        train_list.append(i)

print(len(val_list))
print(len(train_list))

os.makedirs("data/ImageNet/Obj512_all", exist_ok=True)

with open('data/ImageNet/Obj512_all/val.txt', 'w') as f:
    for i in val_list:
        f.write(i+'\n')

with open('data/ImageNet/Obj512_all/train.txt', 'w') as f:
    for i in train_list:
        f.write(i+'\n')

all_list = val_list + train_list
random.shuffle(all_list)

with open('data/ImageNet/Obj512_all/all.txt', 'w') as f:
    for i in all_list:
        f.write(i+'\n')
