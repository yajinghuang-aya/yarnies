import pandas as pd
import numpy as np

fav=pd.read_csv("user_fav_pat_nyc_all.csv",header=0,sep="^")
fav.rename(columns={"Unnamed: 0":"pattern_id"},inplace=True)
pat_fav_list=(fav).sort_values(['time']).drop_duplicates()

category_df=pd.read_csv("pattern_category_all.csv",header=0,sep="^")

category_df=category.drop(columns='Unnamed: 0')
category_df.rename(columns={'category ':'category'})
fav_cat=(pat_fav_list.merge(category,how='inner',on='pattern_id')).dropna(inplace=True)
fav_cat.sort_values(['time'],inplace=True)

top=["coat","jacket","sweater","cardigan",
     "pullover","tops","sleeveless top",
     "strapless top","tee","vest"]


fav_cat_top=pd.DataFrame(columns=fav_cat.columns)
for index,row in fav_cat_recent.iterrows():
    for t in top: 
        if t in row['category'].lower():
            fav_cat_top=fav_cat_top.append(row,ignore_index=True)
            continue

fav_cat_top_2=fav_cat_top.drop(columns='username')

fav_cat_top_2.to_csv("pat_time_count_nyc_all.csv")