import numpy as ny  
import pandas as pd 
from model import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#rank=pd.read_csv("color_rank_136.csv",header=None)

#c=str(rank[0][9])



def get_dataframe_for_forecast(x_week,y,column='h0'):
	feature=x_week.copy()
	feature=feature[['time',column]]
	feature[column]=x_week[column]/x_week['tot']
	feature[column]=feature[column].rolling(4).mean()


	colname=['month','y']
	colname.extend([str(i) for i in range(8)])
	data=pd.DataFrame(columns=colname)
	data['month']=y.time[14:]
	data.reset_index(inplace=True)

	
    month=pd.to_datetime(data['month'].iloc[-1])

    start=month - pd.DateOffset(months=1)
    end=month
    month_predict=month+pd.DateOffset(months=1)
    #print(start,end)
    xs=feature[pd.to_datetime(feature['time'])>=start]
    xs=xs[pd.to_datetime(xs['time'])<end].iloc[-8:]
    xs.reset_index(inplace=True)
    

    for j in range(8):
        data[str(j)].iloc[i]=xs.iloc[j][column]


	return data


columns=['tot']
columns.extend(['h'+str(i) for i in range(6)])
columns.extend(['s'+str(i) for i in range(5)])
columns.extend(['l'+str(i) for i in range(5)])

dat_marginal=pd.read_csv("pop_color_hsl_marginal_weekly.csv")
dat_marginal_monthly=pd.read_csv("pop_color_hsl_marginal_monthly.csv")

for i in columns[1:]:
	get_dataframe_for_model(dat_marginal,dat_marginal_monthly,column=i)
columns=['tot']
columns.extend(['h'+str(i) for i in range(6)])
columns.extend(['s'+str(i) for i in range(5)])
columns.extend(['l'+str(i) for i in range(5)])
dat_rf=pd.read_csv("color_feature_hsl/color_weekly_feature_to_month_h0.csv")
time=dat_rf['month']

for i in columns[1:]:

	dat_rf=pd.read_csv("color_feature_hsl/color_weekly_feature_to_month_"+i+".csv")
	lenth=dat_rf.shape[0]
	print(lenth)
	x=dat_rf.drop(columns=['Unnamed: 0','index','y']).copy()
	y=dat_rf['y'].copy()
	x['year']=x['month'].apply(lambda x: int(x[:4]))
	x['m']=x['month'].apply(lambda x: int(x[-2:]))
	x=x.drop(columns='month'

	model=random_forest(x,y)

	preds_test,r2=model_predection(model,X_valid,y_valid)

	accuracy=forecast_accuracy(preds_test, y_valid)
	print('cat= ',i)
	print(accuracy)

	divide=80
	preds_train = model.predict(X_train)
#r2 = r2_score(y_train, preds)
	fig, ax = plt.subplots(figsize=(8,5))

	ax.plot(pd.to_datetime(time[:divide]),y_train*100,color='orange',label='training data')
	ax.plot(pd.to_datetime(time[:divide]),preds_train*100,color='g',label='model')
	ax.plot(pd.to_datetime(time[83:100]),y_valid*100,color='orange',ls=":",label='test data')
	ax.plot(pd.to_datetime(time[83:100]),preds_test*100,color='g')
	ax.set_title("Category "+i.upper())
	ax.set_xlabel('year')
	ax.set_ylabel('% popularity')
	ax.legend(loc="upper left")
	years = mdates.YearLocator()
	ax.xaxis.set_major_locator(years)
	plt.savefig("color_feature_hsl/dat_fit_"+i+".png")