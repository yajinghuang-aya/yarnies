import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, deltaE_cie76,lab2rgb


dat=pd.read_csv("project_pop_color_200.csv")

dat.drop(columns=["Unnamed: 0","Unnamed: 0.1"],inplace=True)
dat.fillna(value=0,inplace=True)

palette=pd.read_csv("rgb.txt",header=0,sep='#')
for index, row in palette.iterrows():
    row['colorname']=row['colorname'][:-1]
    row['hexcode']='#'+row['hexcode'][:-1]
    

hexcode=palette['hexcode']
colorname=palette['colorname']
def hex2rgb(hexcode):
    h= hexcode.lstrip('#')
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)],dtype='uint8')

hexcode_lab=np.empty((0,3),dtype='uint8')
for p in hexcode:
    p_lab=(rgb2lab(hex2rgb(p))).reshape((1,3))
    hexcode_lab=np.append(hexcode_lab,p_lab,axis=0)
    
colorcode=pd.read_excel("color136.xlsx")
colorcode=colorcode.rename(columns={'Hex Equivalent':'colorname','Unnamed: 1':'hexcode'})
colorcode.iloc[8]["hexcode"]='#F0F8FF'
color_136=np.empty((136,3))
for i in range(136):
    color_136[i]=hex2rgb(colorcode['hexcode'][i])
color_136=color_136.astype('uint8')

color_949_to_136=[]
for i in range(949):
    color_949_to_136.append(np.argmin(deltaE_cie76(hexcode_lab[i],rgb2lab(color_136))))

start=pd.to_datetime('2010/12')
end=pd.to_datetime('2020/04')
x=dat[pd.to_datetime(dat['time'])>start].copy()
x=x[pd.to_datetime(x['time'])<end]

x=x.sort_values('time')
x=x.fillna(value=0)

x_day=x.groupby('time').sum()

column=['time']
column.extend([i for i in range(136)])
color_time_136=pd.DataFrame(columns=column)
color_time_136['time']=x_day.index
color_time_136.fillna(value=0,inplace=True)  #important!!!!

for i in range(136):
    for j in range(949):
        if color_949_to_136[j]==i:
            #print(j,i)
            color_time_136[i]=np.asarray(color_time_136[i])+np.asarray(x_day[str(j)])

color_time_136.fillna(value=0,inplace=True)
color_time_136['tot']=color_time_136.iloc[:, 1:].sum(axis=1)
print(color_time_136.columns)

color_time_136.to_csv("project_pop_color136.csv")
