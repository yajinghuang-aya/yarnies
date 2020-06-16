import time
import random
import pandas as pd
import numpy as np
import requests
import json

user="2615aa2d9a9c4e1d63a9b961b018cc68"
#pwd="gXO/8avXOh1hYe8F3ocOcSSybfDtOIQE1O8qGwVW"
pwd="q6x-RFUjwvc_rdhGyCLeKxkSrDiHNcA07e0pL6EO"

def get_api_result(url):
    result = requests.get(url,auth=requests.auth.HTTPBasicAuth(user, pwd)).json()
    return result

job_id=0
id_per_job=90#000
start=job_id*id_per_job
end=(job_id+1)*id_per_job
user_id_all=np.load('user_id_random.npy')
user_id=user_id_all[start:end]


user_df=[]
start_time=time.time()
for i in user_id:
    
    series={}
    series['user_id']=i
    url="https://api.ravelry.com/people/"+str(i)+".json"
    #print(url)
    try:
        user_data=get_api_result(url)['user']
    except :
        continue
    user_name=user_data['username']
    location=user_data['location']
    series['user_name']=user_name
    series['location']=location
    user_df.append(series)

    
end_time=time.time()
print(end_time-start_time)

df=pd.DataFrame(user_df)
df=df.set_index('user_id')
outfile='users_'+str(job_id)+'.pkl'
df.to_pickle(outfile)
#df.to_pickle("./userID_5000.pkl")