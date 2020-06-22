import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, deltaE_cie76,lab2rgb


def get_dataframe_for_model(x_week,y,column='h0'):
	feature=x_week.copy()
	feature=feature[['time',column]]
	feature[column]=x_week[column]/x_week['tot']
	feature[column]=feature[column].rolling(4).mean()


	colname=['month','y']
	colname.extend([str(i) for i in range(12)])
	data=pd.DataFrame(columns=colname)
	data['month']=y.time[14:]
	data.reset_index(inplace=True)

	for i in range(data.shape[0]):#[:5]:
	    month=pd.to_datetime(data['month'].iloc[i])

	    start=month - pd.DateOffset(months=4)
	    end=month- pd.DateOffset(months=1)
	    #print(start,end)
	    xs=feature[pd.to_datetime(feature['time'])>=start]
	    xs=xs[pd.to_datetime(xs['time'])<=end].iloc[-12:]
	    xs.reset_index(inplace=True)
	    #print(xs)
	    #print('')
	    y_current=y[y['time']==data['month'].iloc[i]]
	    
	    #print('')
	    #print(y_current['41']/y_current['tot'])
	    #print(data['y'].iloc[i])
	    data['y'].iloc[i]=(y_current[column]/y_current['tot']).values[0]
	    for j in range(12):
	        data[str(j)].iloc[i]=xs.iloc[j][column]

	#data['y']=(y['41']/y['tot'])[60:]
	data.to_csv("color_feature_hsl/color_weekly_feature_to_month_"+column+".csv")

	return data


# for 1D: H, S, or L
columns=['tot']
columns.extend(['h'+str(i) for i in range(6)])
columns.extend(['s'+str(i) for i in range(5)])
columns.extend(['l'+str(i) for i in range(5)])

dat_marginal=pd.read_csv("pop_color_hsl_marginal_weekly.csv")
dat_marginal_monthly=pd.read_csv("pop_color_hsl_marginal_monthly.csv")


# for 2D: SxL
#columns=['tot']
#columns.extend(['s'+str(i)+'l'+str(j) for i in range(5) for j in range(5)])

#dat_marginal=pd.read_csv("pop_color_hsl_marginal2D_weekly.csv")
#dat_marginal_monthly=pd.read_csv("pop_color_hsl_marginal2D_monthly.csv")


for i in columns[1:]:
	get_dataframe_for_model(dat_marginal,dat_marginal_monthly,column=i)




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





