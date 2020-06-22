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

columns.extend(['s'+str(i)+'l'+str(j) for i in range(5) for j in range(5)])

dat_rf=pd.read_csv("color_feature_hsl/color_weekly_feature_to_month_h0.csv")
time=dat_rf['month']

for i in columns[1:]:

	(X_train,y_train,X_valid,y_valid) = prepare_data(color_code=i)

	model=random_forest(X_train,y_train)

	preds_test,r2=model_predection(model,X_valid,y_valid)

	accuracy=forecast_accuracy(preds_test, y_valid)
	print('cat= ',i)
	print(accuracy)
	size=dat_rf.shape[0]
	divide=(int(size*0.9))
	preds_train = model.predict(X_train)
#r2 = r2_score(y_train, preds)

	from colour import hsl2rgb

	time_save=time[:divide].append(time[divide:])
	data_save=np.append(y_train,y_valid)
	model_save=np.append(preds_train,preds_test)
	#dict={'time':time_save,'data':data_save,'model':model_save}
	#df_save=pd.DataFrame(data=dict)
	#df_save.to_csv("color_feature_hsl/model_fit_"+i+".csv")

	fig, ax = plt.subplots(figsize=(8,5))

	ax.plot(pd.to_datetime(time[:divide]),y_train*100,color=hsl2rgb([0.6,0.5,70/100.]),lw=3,
		label='data')
	ax.plot(pd.to_datetime(time[:divide]),preds_train*100,color=hsl2rgb([0.6,0.5,10/100.]),lw=2,ls=":",label='model')
	ax.plot(pd.to_datetime(time[divide:size]),y_valid*100,color=hsl2rgb([0.6,0.5,70/100.]),lw=3)#,ls=":")#,label='test data')
	ax.plot(pd.to_datetime(time[divide:size]),preds_test*100,color=hsl2rgb([0.6,0.5,10/100.]),lw=2,ls=":")
	ax.set_title("Category "+i.upper())
	ax.set_xlabel('year',fontsize=15)
	ax.set_ylabel('% popularity',fontsize=15)
	ax.legend(loc="upper right",frameon=False)
	years = mdates.YearLocator()
	#ax.xaxis.set_major_locator(years)
	plt.savefig("color_feature_hsl/dat_fit_"+i+".png")



