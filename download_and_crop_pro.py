import wget
import cv2
from object_detection.utils import label_map_util  #packege from https://github.com/tensorflow/models.git 
import pandas as pd  
import numpy as np  
import time
from os import path 

from scraping_util import *
from crop_util import *


label=[2,68,99,128]
#top=["coat","jacket","sweater","cardigan","pullover","tops","sleeveless top","strapless top","tee","vest"]

job_id=0

infile='project_csv/project_nyc_'+str(job_id)+'.csv'
data=pd.read_csv(infile,header=0,sep="^")

#print(data.shape)
model_name = 'faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28'
model=load_model(model_name) #from crop_util.py
print('model loaded')

start_time=time.time()
for index, row in data[10:20].iterrows():
	item_id=row['project_id']
	#print(item_id,cat)
	print(item_id)
	outpath='crop_image/'+str(item_id)+'.jpg'
	if path.exists(outpath):
		print('crop image already exists')
		continue
	
	image_url=row['first_photo']

	try:
		dir_path='image_download'
		filename = wget.download(image_url,out=dir_path)
	except:
		print('image download error')
		continue

	image_np = (cv2.imread(filename))#.astype(int)
	
	crop_image(image_np,model,outpath)  #crop and save images
	

end_time=time.time()
print('time',end_time-start_time)


