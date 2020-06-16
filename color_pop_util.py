import cv2
import wget
from color_util import *
from color_cluster import* 
from skimage.color import rgb2lab, deltaE_cie76,lab2rgb
from skimage import io
import numpy as np
import pandas as pd
import time


palette=pd.read_csv("rgb.txt",header=0,sep='#')
for index, row in palette.iterrows():
    row['colorname']=row['colorname'][:-1]
    row['hexcode']='#'+row['hexcode'][:-1]

hexcode=palette['hexcode']
colorname=palette['colorname']


def hex2rgb(hex):
    h= hex.lstrip('#')
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)],dtype='uint8')
    
hexcode_lab=np.empty((0,3),dtype='uint8')
for p in hexcode:
    p_lab=(rgb2lab(hex2rgb(p))).reshape((1,3))
    hexcode_lab=np.append(hexcode_lab,p_lab,axis=0)




def image_pop_color(image,image_id):

    image=prepare_image(image).astype('uint8')
    image_lab=rgb2lab(image)
    num_pix=image.shape[0]
    image_color_index=[]

    start_time=time.time()
    for i in range(num_pix):
        image_color_index.append(
            np.argmin(deltaE_cie76(image_lab[i],hexcode_lab)))

    (unique, counts)=np.unique(image_color_index, return_counts=True)
    frequency={u:c for(u,c) in zip(unique, counts)}
    frequency['pattern_id']=image_id
    end_time=time.time()
    print('time',end_time-start_time)
    return frequency



