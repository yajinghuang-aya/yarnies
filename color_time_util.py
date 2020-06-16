from color_cluster import *
import pandas as pd 
import numpy as np
from os import path
import cv2

def color_pattern_clustering(pattern_id,folder="crop_image/",outfolder="cluster_dict/"):
    filepath=folder+str(pattern_id)+'.jpg'
    outfile=outfolder+str(pattern_id)+'.txt'
    print(pattern_id, "is running")
    if path.exists(filepath) and (not path.exists(outfile)):
        image=cv2.imread(filepath)
        image=prepare_image(image)

        color=kmeans_op(image)

        #if color:
        np.savetxt(outfile,color)
        print(pattern_id,"write to dict")
        #else:
         #   print("no good elbow for ",pattern_id)
         #   file_object = open('cluster_error_patID.txt', 'a')
         #   file_object.write(str(pattern_id))
         #   file_object.close()
    else:
        print('image not exists, or txt already exists')
    return None
        
def get_color_data_from_clusters(df,month):
    output={}
    df = df[df.time==month]
    weights=[]
    colors = np.empty((0,4), int)
    color_dict={}
    for index, row in df.iterrows():
        pattern_id=row['pattern_id']
        fname='cluster_dict/'+str(pattern_id)+'.txt'

        if path.exists(fname):
            if pattern_id not in color_dict.keys():
                color_dict[pattern_id]=np.loadtxt(fname)
            
            color =color_dict[pattern_id]
            color[:,3]=color[:,3]*row['count']
            colors = np.append(colors, color, axis=0)

    output[month]=colors
    
    return output

def color_vs_month(df):
    color_dict_months={}
    months=df.time.unique()
    for m in months:
        color_dict_months.update(get_color_data_from_clusters(df,m))
    return color_dict_months

def df_pattern_to_color(df):
    ids=df.pattern_id.unique()
    for i in ids:
        color_pattern_clustering(i)
    print('df pattern to color done')
    

def get_df_sample(df,frac=0.1):
    months=df.time.unique()
    df_out=df[df.time==months[0]].sample(frac=frac,random_state=1)
    for m in months[1:]:
        d=df[df.time==m].sample(frac = frac,random_state=1)
        if not d.empty:
            df_out=df_out.append(d,ignore_index=True)
    return df_out
