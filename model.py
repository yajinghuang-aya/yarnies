import numpy as ny  
import pandas as pd 
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor



def prepare_data(color_code='41'):

	dat_rf=pd.read_csv("color_weekly_feature_to_month_"+color_code+".csv")
	x=dat_rf.drop(columns=['Unnamed: 0','index','y']).copy()
	y=dat_rf['y'].copy()
	x['year']=x['month'].apply(lambda x: int(x[:4]))
	x['m']=x['month'].apply(lambda x: int(x[-2:]))
	x=x.drop(columns='month')

	X_train=x.iloc[:77]
	y_train=y.iloc[:77]
	X_valid=x.iloc[80:]
	y_valid=y.iloc[80:]


	return  (X_train,y_train,X_valid,y_valid)



def random_forest(n_estimators=11,random_state=1):
	model = RandomForestRegressor(n_estimators=n_estimators,min_samples_leaf=2, random_state=random_state)
	model.fit(X_train, y_train)

	return model

def model_predection(model,X_valid,y_valid)
	preds = model.predict(X_valid)
	r2 = r2_score(y_valid, preds)
	print('R2 score:', r2)
	return (preds,r2)

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

    
forecast_accuracy(preds,y_valid)