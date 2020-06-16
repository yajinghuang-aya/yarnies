import pandas as pd
import numpy as np
import requests
#from scraping_util import *

job_id=0
print('job_id',job_id)

user="2615aa2d9a9c4e1d63a9b961b018cc68"
pwd="q6x-RFUjwvc_rdhGyCLeKxkSrDiHNcA07e0pL6EO"
url_ravelry="https://api.ravelry.com/"

url=url_ravelry+'projects/search.json?'+ \
'pc=coat|sweater|tops|vest&photo=yes&within=40miles&sort=started'

page_tot=1188
pages_all=np.arange(1,1189)
pages=pages_all[pages_all%200==job_id]


data=[]

for page in pages:
	print("page:",page)
	try:
		output=requests.get(url,auth=requests.auth.HTTPBasicAuth(user, pwd),data={'page_size': 48,'page': page}).json()
	except:
		print('page loading error: <',url,'> page ',page)
		continue
	data.extend(output['projects'])

filename='project_csv/project_nyc_'+str(job_id)+'.csv'
column_names = ["project_id",'time',"username","first_photo"]
df_pro=pd.DataFrame(columns = column_names)
df_pro.to_csv(filename,sep="^")


for p in data:
	
	project_info=[p['id'],p['created_at'][:10],p['user']['username'],p['first_photo']['medium_url']]
	project_df={n:pro for n,pro in zip(column_names,project_info)}
	current_pro_df=pd.DataFrame(project_df,index=[p['id']])

	f = open(filename, 'a')
	current_pro_df.to_csv(f,sep="^", header = False)
	f.close() 

	print(p['id'],' save to csv')       
