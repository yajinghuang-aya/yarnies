import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, deltaE_cie76,lab2rgb


def get_dataframe_for_model(x_week,y,column='h0'):
	feature=x_week.copy()
	feature=feature[['time',column]]
	feature[column]=x_week[column]/x_week['tot']
	feature[column]=feature[column].rolling(4).mean()


	colname=['month']
	colname.extend([str(i) for i in range(12)])
	data=pd.DataFrame(columns=colname)
	#print(y['time'])
	data['month']=y['time'][114]

	data.reset_index(inplace=True)

	#for i in range(data.shape[0]):#[:5]:)
	xs=feature
	xs=xs.iloc[-12:]
	xs.reset_index(inplace=True)
	    #print(xs)
	print(xs)
	print(data)
	for j in range(12):
	    data[str(j)].iloc[0]=xs.iloc[j][column]

	#data['y']=(y['41']/y['tot'])[60:]
	data.to_csv("color_feature_hsl/color_weekly_last_feature_"+column+".csv")

	return data


columns=['tot']
#columns.extend(['h'+str(i) for i in range(6)])
#columns.extend(['s'+str(i) for i in range(5)])
#columns.extend(['l'+str(i) for i in range(5)])
columns.extend(['s'+str(i)+'l'+str(j) for i in range(5) for j in range(5)])

dat_marginal2D=pd.read_csv("pop_color_hsl_marginal2D_weekly.csv")
dat_marginal2D_monthly=pd.read_csv("pop_color_hsl_marginal2D_monthly.csv")

#for i in columns[1:]:
#	get_dataframe_for_model(dat_marginal2D,dat_marginal2D_monthly,column=i)

get_dataframe_for_model(dat_marginal2D,dat_marginal2D_monthly,column='s0l2')


#x_week=pd.read_csv("project_color_pop_weekly.csv")
#y=pd.read_csv("pro_color_time_for_ts.csv").iloc[:-3]
#y.reset_index(inplace=True)


#color_time_136=pd.read_csv("project_pop_color136.csv")
#color_time_136.drop(columns=['Unnamed: 0'],inplace=True)

#x_week=color_time_136.copy()
#x_week['time']=pd.to_datetime(x_week['time']) - pd.to_timedelta(7, unit='d')
#x_week = x_week.groupby(pd.Grouper(key='time', freq='W-MON')).sum().reset_index().sort_values('time')

#tot=x_week.sum(axis=0)
#color_count=tot
#color_count=color_count/color_count['tot']
#rank=color_count.sort_values(ascending=False)[1:]
#color_rank_index=rank.keys()



#c='41'

#for c in color_rank_index[:25]:

#	get_dataframe_for_model(x_week,y,column=c)





