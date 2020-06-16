import numpy as np
import pandas as pd 
import os
import six.moves.urllib as urllib
import sys
import pickle
import time

#import wget
#import cv2

#from matplotlib import pyplot as plt
#from object_detection.utils import ops as utils_ops
#from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util
#from PIL import Image
#import tensorflow as tf
from scraping_util import *

job_id=0

file_input="pat_fav_list_nyc_unique.csv"
file_output="output/pattern_category_"+str(job_id)+".csv"

header="pattern_id^category\n"
text_file = open(file_output, "w")
text_file.write(header)
text_file.close()

data_all=pd.read_csv(file_input,header=0,sep="^")
data=data_all[data_all.index%100==job_id]

url_ravelry="https://api.ravelry.com/"

#labels={'clothing':2,'Scarf'}
start_time=time.time()

for index, row in data.iterrows():
	item_id=row['pattern_id']

	cat_current=str(item_id)+'^'
	cat='NaN'
	try:
		pat=get_text_onepage(url_ravelry+'patterns/'+str(item_id)+'.json')
	except:
		print('pattern loading error',item_id)
		continue

	if pat['pattern']['pattern_categories']:
		cat=''
		for category in pat['pattern']['pattern_categories']:
			pat_name=category['name']
			if pat_name == "Other":
				pat_name=category['parent']['name']

			cat=cat+'_'+pat_name

	cat_current+=cat 

	text_file = open(file_output, "a")
	text_file.write(cat_current+'\n')
	text_file.close()

end_time=time.time()
print(end_time-start_time)

