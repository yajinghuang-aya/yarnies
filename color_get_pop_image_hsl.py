from color_pop_util import *
import numpy as np
import pandas as pd
from os import path
import cv2

#divde jobs for parallel running
job_id=0
print('job_id ',job_id)


df_ids=pd.read_csv("pat_fav_list_nyc_unique.csv",header=0,sep="^")

df_ids=df_ids[df_ids.index%200==job_id]

ids=df_ids['pattern_id']

column=['pattern_id']
column.extend([i for i in range(949)])
image_color_df=pd.DataFrame(columns=column)

for i in ids:
	print('pattern_id ',i)
	fname='crop_image/'+str(i)+'.jpg'
	if path.exists(fname):
		image=cv2.imread(fname)
		frequency=image_pop_color(image,image_id)
		image_color_df=image_color_df.append(frequency,ignore_index=True)
	else:
		print("file not exists")

image_color_df.to_csv("image_pop_color/pop_color_"+str(job_id)+'.csv')



