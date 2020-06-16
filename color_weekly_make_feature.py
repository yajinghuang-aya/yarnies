import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, deltaE_cie76,lab2rgb

x_week=pd.read_csv("project_color_pop_weekly.csv")
y=pd.read_csv("pro_color_time_for_ts.csv").iloc[:-3]
y.reset_index(inplace=True)
feature=x_week.copy()
feature=feature[['time','41']]
feature['41']=x_week['41']/x_week['tot']
feature['41']=feature['41'].rolling(4).mean()


colname=['month','y']
colname.extend([str(i) for i in range(8)])
data=pd.DataFrame(columns=colname)
data['month']=y.time[60:]
data.reset_index(inplace=True)

for i in range(data.shape[0]):#[:5]:
    month=pd.to_datetime(data['month'].iloc[i])

    start=month - pd.DateOffset(months=2)
    end=month
    #print(start,end)
    xs=feature[pd.to_datetime(feature['time'])>=start]
    xs=xs[pd.to_datetime(xs['time'])<end].iloc[-8:]
    xs.reset_index(inplace=True)
    #print(xs)
    #print('')
    y_current=y[y['time']==data['month'].iloc[i]]
    
    #print('')
    #print(y_current['41']/y_current['tot'])
    #print(data['y'].iloc[i])
    data['y'].iloc[i]=(y_current['41']/y_current['tot']).values[0]
    for j in range(8):
        data[str(j)].iloc[i]=xs.iloc[j]['41']

#data['y']=(y['41']/y['tot'])[60:]
data.to_csv("color_weekly_feature_to_month_41.csv")




