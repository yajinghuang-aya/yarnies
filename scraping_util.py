import time
import random
import pandas as pd
import numpy as np
import requests
import json

user="2615aa2d9a9c4e1d63a9b961b018cc68"
pwd="q6x-RFUjwvc_rdhGyCLeKxkSrDiHNcA07e0pL6EO"
url_ravelry="https://api.ravelry.com/"

# get data, scrolling pages

def get_text_allpage(url,content='favorites'):
    output=requests.get(url,auth=requests.auth.HTTPBasicAuth(user, pwd),data={'page_size': 40}).json()
    num_pages=output['paginator']['last_page']
    data=output[content]
    for page in range(2,num_pages+1):
        try:
            output=requests.get(url,auth=requests.auth.HTTPBasicAuth(user, pwd),data={'page_size': 40,'page': page}).json()
        except:
            print('page loading error: <',url,'> page ',page)
            continue
        data.extend(output[content])
        #print(page)
        
    return data

def get_text_onepage(url):

    result = requests.get(url,auth=requests.auth.HTTPBasicAuth(user, pwd)).json()
    return result



def favorites_per_user(user_dat,filename):
    url_fav_list=url_ravelry+'people/'+user_dat.user_name+'/favorites/list.json'
    fav_list=get_text_allpage(url_fav_list,content='favorites')



    column_names = ["user_id", "username", "location",'time',"yarn_id",
              "yarn_permalink","yarn_weight"]
    #df_fav=pd.DataFrame(columns = column_names)
    #df_fav=pd.DataFrame(current_user)

    for f in fav_list:
        current_fav={"user_id":user_dat.user_id, "username":user_dat.user_name,
              "location":user_dat.location}
        current_fav['time']=f['created_at']
        fav_type=f['type']

        if fav_type == 'pattern':
            content=f['favorited']
            item_id=content['id']
            item_url=url_ravelry+'patterns/'+str(item_id)+'.json'
            item_link=content['permalink']
            try:
                item=get_text_onepage(item_url)
            except:
                print("pattern loading error: ",item_link)
                file_object = open(filename+'_error', 'a')
                file_object.write(item_link+' \n')
                file_object.close()
                continue
            #print('ha')

            if item['pattern']['packs'] and \
                item['pattern']['packs'][0]['yarn_id']: 
                # if pattern specifies yarn
                #print('yarn')
                yarn_packs=item['pattern']['packs']
                for y in yarn_packs:
                    if y['yarn_id'] :
                        current_fav['yarn_id']=y['yarn_id']
                    if y['yarn']['permalink']:
                        current_fav['yarn_permalink']=y['yarn']['permalink']
                    if y['yarn_weight']:
                        current_fav['yarn_weight']=y['yarn_weight']['name']

                    #df_fav=df_fav.append(current_fav,ignore_index=True)  
                    index=str(user_dat.user_id)+' '+current_fav['time']
                    current_save=pd.DataFrame(current_fav,
                                             index=[index])
                    #add to file
                    f = open(filename, 'a')
                    current_save.to_csv(f,sep="^", header = False)
                    f.close()


def favorites_permalink_per_user(user_dat,filename):
    url_fav_list=url_ravelry+'people/'+user_dat.user_name+'/favorites/list.json'
    fav_list=get_text_allpage(url_fav_list,content='favorites')


    for f in fav_list:
        current_fav={"username":user_dat.user_name}
              #"location":user_dat.location}
        current_fav['time']=f['created_at'][:7]
        fav_type=f['type']

        if fav_type == 'pattern':
            content=f['favorited']
            current_fav['permalink']=content['permalink']
            if content['first_photo']:
                current_fav['first_photo']=\
                    content['first_photo']['medium_url']
            else: current_fav['first_photo']=np.nan

            index=content['id']
            current_save=pd.DataFrame(current_fav,
                                     index=[index])
            #add to file
            f = open(filename, 'a')
            current_save.to_csv(f,sep="^", header = False)
            f.close()

