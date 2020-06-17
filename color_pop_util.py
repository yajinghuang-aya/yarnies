import cv2
import wget
from color_util import *
from color_cluster import* 
from skimage.color import rgb2lab, deltaE_cie76,lab2rgb
from skimage import io
import numpy as np
import pandas as pd
import time
from colour import Color


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
    frequency['project_id']=image_id
    end_time=time.time()
    print('time',end_time-start_time)
    return frequency

def rgb2hsl(pix):
    pix=np.asarray(pix)/255.
    pixcolor=Color(rgb=pix)
    hsl=pixcolor.hsl
    color_hsl=np.zeros(3)
    color_hsl[0]=hsl[0]*360
    color_hsl[1]=hsl[1]*100
    color_hsl[2]=hsl[2]*100

    return color_hsl

def image_pop_color_hsl(image,image_id,t):

    hue=[[330,30],[30,90],[90,150],[150,210],[210,270],[270,330]]
    satuation=[[0,20],[20,40],[40,60],[60,80],[80,100]]
    lightness=[[0,20],[20,40],[40,60],[60,80],[80,100]]
    hsl_cat=np.zeros((6,5,5))
    image=prepare_image(image).astype('uint8')
    for i in range(image.shape[0]):
        pix_hsl=rgb2hsl(image[i])
        for i in range(1,6):
            if pix_hsl[0]>=hue[i][0] and pix_hsl[0]<hue[i][1]:
                pix_h=i
        if pix_hsl[0]>=330 or pix_hsl[1]<30:
                pix_h=0
        for i in range(5):
            if pix_hsl[1]>=satuation[i][0] and pix_hsl[1]<satuation[i][1]:
                pix_s=i
            if pix_hsl[2]>=lightness[i][0] and pix_hsl[2]<lightness[i][1]:
                pix_l=i
        hsl_cat[pix_h,pix_s,pix_l]+=1

    color_cat_flat=hsl_cat.reshape(150)
    frequency={}
    for i in range(150):
        frequency[i]=color_cat_flat[i]
    frequency['project_id']=image_id
    frequency['time']=t
    return frequency


def oneD_to_3D_index(o):

    i1=o//25
    i2=(o%25)//5
    i3=(o%25)%5
    return i1,i2,i3









