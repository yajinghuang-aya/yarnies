import numpy as ny  
import pandas as pd 
from model import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#rank=pd.read_csv("color_rank_136.csv",header=None)

#c=str(rank[0][9])
columns=['tot']
columns.extend(['h'+str(i) for i in range(6)])
columns.extend(['s'+str(i) for i in range(5)])
columns.extend(['l'+str(i) for i in range(5)])
dat_rf=pd.read_csv("color_feature_hsl/color_weekly_feature_to_month_h0.csv")
time=dat_rf['month']

for i in columns[1:]:

	(X_train,y_train,X_valid,y_valid) = prepare_data(color_code=i)

	model=random_forest(X_train,y_train)

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



