import cv2
import wget
from sklearn_extra.cluster import KMedoids
from PIL import Image
from color_util import *
from color_cluster import *
import time
import pandas as pd 
import numpy as np  
from os import path



job_id=0
df=pd.read_csv("pat_time_count_nyc_sample.csv",header=0,sep="^")
pattern_id=df.pattern_id.unique()
num_id=len(pattern_id)
index=np.arange(num_id).astype('int')
index=index[index%100==job_id][:10]
print(index)
start_time=time.time()
for i in index:
	pat_id=pattern_id[i]
	filename="crop_image/"+str(pat_id)+'.jpg'
	print(filename)
	if path.exists(filename):
		image=cv2.imread(filename)
		image=flip_rgb(image)
		#if image:
		print('clustering...')
		try:
			color=color_cluster(image)
		except:
			print('color cluster failed',pat_id)
			continue
		print('writing to file')
		outfile="cluster_df/"+str(pat_id)+".csv"
		color.to_csv(outfile,sep="^")

	else:
		print(pat_id,"no jpg")

end_time=time.time()
print('time',end_time-start_time)
