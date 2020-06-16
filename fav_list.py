import time
import random
import pandas as pd
import numpy as np
import requests
import json
from scraping_util import *

user="2615aa2d9a9c4e1d63a9b961b018cc68"
pwd="q6x-RFUjwvc_rdhGyCLeKxkSrDiHNcA07e0pL6EO"
url_ravelry="https://api.ravelry.com/"


austin_name=['austin, texas','austin,texas',
             'austin,tx','austin, tx','austin']
nyc_name=['new york city,ny','new york city, ny','new york, ny',
          'new york,ny','new york,new york','new york ,new york',
          'new york city','new york']

job_id=0
datafile='user_id/user_'+str(job_id)+'.csv'

user_all=pd.read_csv(datafile,header=0,sep="^")

user_sub=user_all[user_all['location'].str.lower().isin(nyc_name)]
user=user_sub.iloc[:5]
#filename='user_fav_nyc_'+str(job_id)+'.csv'
filename='user_fav.csv'
for index, row in user.iterrows():
    try:
    	favorites_per_user(row,filename)
    except:
        print(row.username,'cannot load favorites')
        continue
    print(row.user_id,row.user_name,'fav loaded')



