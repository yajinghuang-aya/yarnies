import wget
import cv2
from object_detection.utils import label_map_util
import pandas as pd  
import numpy as np  
import time
from os import path
from scraping_util import *
from crop_util import *
label=[2,68,99,128]

top=["coat","jacket","sweater","cardigan","pullover","tops","sleeveless top","strapless top","tee","vest"]

job_id=0
print('job id: ',job_id)
dir_path='image_download/'

data_all=pd.read_csv("pat_photo_cat_nyc.csv",header=0,sep="^")
data=data_all[data_all.index%200==job_id]
#print(data.shape)
data=data.dropna(inplace=True)
model_name = 'faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28'
model=load_model(model_name)
print('model loaded')

start_time=time.time()
for index, row in data.iterrows():
	item_id=row['pattern_id']
	cat=row['category'].lower()

	outpath='crop_image/'+str(item_id)+'.jpg'
	if path.exists(outpath):
		print('crop image already exists')
		continue

	
	#print(item_id,cat)
	for i in top:
		if i in cat:
			print(i,row['pattern_permelink'])
			image_url=row['first_photo']

			try:
				filename = wget.download(image_url,out=dir_path)
			except:
				print('image download error')
				continue

			image_np = (cv2.imread(filename))#.astype(int)
			
			
			crop_image(image_np,model,outpath)
			continue

end_time=time.time()
print('time',end_time-start_time)


