import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

color_time_136=pd.read_csv("project_pop_color136.csv")
color_time_136.drop(columns=['Unnamed: 0'],inplace=True)

x_week=color_time_136.copy()
x_week['time']=pd.to_datetime(x_week['time']) - pd.to_timedelta(7, unit='d')
x_week = x_week.groupby(pd.Grouper(key='time', freq='W-MON')).sum().reset_index().sort_values('time')

tot=x_week.sum(axis=0)
color_count=tot
color_count=color_count/color_count['tot']
rank=color_count.sort_values(ascending=False)[1:]
color_rank_index=rank.keys()

#color_rank=pd.read_csv("color_rank_136.csv")

num_pre=16
num_roll=6

for c in color_rank_index[:25]:
    col=(x_week[str(c)]/x_week['tot']).rolling(num_roll).mean()[num_roll-1:]
    data=pd.DataFrame(columns=['year','week','min','max','mean','std'])
    data['week']=x_week['time'].dt.week[num_roll-1+num_pre:]
    data.reset_index(inplace=True)
    data=data.drop(columns=['index'])
    #print(data.head())


    for i in range(data.shape[0]):
        data['year'].iloc[i]=(x_week['time'][num_roll-1+num_pre+i].year )#.copy()
        data['week'].iloc[i]=(x_week['time'].copy().dt.week[num_roll-1+num_pre+i])#.copy()
        stat=col[i:i+num_roll].describe()
        #print(stat['min'])
        data['min'].iloc[i]=stat['min']
        #print(data.iloc[i]['min'])
        data['max'].iloc[i]=stat['max']
        data['mean'].iloc[i]=stat['mean']
        data['std'].iloc[i]=stat['std']

    #print(data)   

    data.to_csv("color_rank_features/color_pop_weekly_features_"+str(c)+".csv")

