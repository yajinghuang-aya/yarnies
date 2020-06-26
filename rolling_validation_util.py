#from pmdarima.arima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE #Recursive Feature Elimination
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima

def rolling_rf_data_pre(data,col='s0l2'):  #data weekly

    
    for j in range(1,len(data.columns)-1):
        data.iloc[:,j]=data.iloc[:,j]/data.iloc[:,-1]
    
    data['time']=pd.to_datetime(data['time'])
    data_month=data.groupby(pd.Grouper(key='time',freq='M')) \
                .mean().reset_index().sort_values('time')
    
    #data_month=data_month.set_index('time')
   
    
    
    lags = pd.DataFrame()
    #lags['time']=data_month['time'].copy()
    for i in range(12,3,-1):
        lags['t-'+str(i)] = data_month[col].shift(i)
    lags['t'] = data_month[col].values
    lags = lags[15:]


    array = lags.values
    X = array[:,0:-1]
    y = array[:,-1]
   
    #print(X)
    rfe = RFE(RandomForestRegressor(n_estimators=15, random_state=0), 8)
    fit = rfe.fit(X, y)
    names = lags.columns
    columns_pre=[]
    for i in range(len(fit.support_)):
        if fit.support_[i]:
            columns_pre.append(names[i])

    #print("Columns with predictive power:", columns_pre )
    
    df_forecasting_col=['time',col]
    df_forecasting_col.extend(['week -'+str(i) for i in range(1,9)])
    df_forecasting=pd.DataFrame(columns=df_forecasting_col)
    df_forecasting[col]=data_month[col].copy()
   
    df_forecasting['time']=data_month['time'].copy()
    for i in columns_pre:
        df_forecasting[i] = df_forecasting[col].shift(int(i[2:]))
    for i in range(6,3,-1):
        df_forecasting['t-'+str(i)] = df_forecasting[col].shift(i)

    
    for row in range(df_forecasting.shape[0]):
        
        data_week_past=(data[data['time']<
                             df_forecasting.iloc[row]['time']-pd.DateOffset(weeks=3)][col]).rolling(6).mean()
        #print(data_week_past.tail())
        
        if data_week_past.shape[0]>9:
            #print('yes')
            for i in range(1,9):
             #   print(data_week_past.iloc[-i])
                df_forecasting.at[row,'week -'+str(i)] = data_week_past.iloc[-i]
    df_forecasting= df_forecasting.dropna()
    df_forecasting.set_index('time',inplace=True)
    return  df_forecasting


def rolling_validation_rf(df_forecasting):
    
    size=df_forecasting.shape[0]
    start_size=int(size*0.6)
    #print(start_size)
    accuracy=[]
    y_test=[]
    y_pred=[]
    pred_train_list=[]
    for roll_divide in range(start_size,size) :
        
        x=df_forecasting.iloc[:roll_divide,1:].copy()
        #print(x.shape)
        y=df_forecasting.iloc[:roll_divide,0].copy()
        x_train, x_valid = x.iloc[:-1], x.iloc[-1:]
        y_train, y_valid = y.iloc[:-1], y.iloc[-1:]
        mdl=RandomForestRegressor(n_estimators=15,random_state=0,min_samples_leaf=2)

        mdl.fit(x_train, y_train)
        pred=mdl.predict(x_valid)
        pred=pd.Series(pred, index=y_valid.index)
        
        accuracy.append(forecast_accuracy(pred.values, y_valid)['mape'])
        y_test.append(y_valid)
        y_pred.append(pred)
        
        pred_train=mdl.predict(x_train)
        #pred_train=pd.Series(pred_train, index=x_train.index)
        pred_train_list.append([pred_train])
        
    return df_forecasting.index[start_size:],y_test,y_pred,accuracy,pred_train_list


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
    #acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse,# 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

def rolling_validation_arima(data,col='s0l2'):
    
    size=data.shape[0]
    start_size=int(size*0.6)
    time=data.index
    
    accuracy=[]
    y_test=[]
    y_pred=[]
    pred_train_list=[]
    for roll_divide in range(start_size,size) :
        y=data.iloc[:roll_divide][col]
        train=y.iloc[:-1]
        test=y.iloc[-1:]
    
        stepwise_model = auto_arima(train, start_p=0, start_q=0, max_p=10, max_q=10, m=12, start_P=0, seasonal=True,
                            d=None,max_d=2, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
        
        stepwise_model.fit(train)
        stepwise_model = auto_arima(train, start_p=0, start_q=0, max_p=10, max_q=10, m=12, start_P=0, seasonal=True,
                            d=None,max_d=2, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)

        stepwise_model.fit(train)
        future_forecast = stepwise_model.predict(n_periods=1)
        accuracy.append(forecast_accuracy(future_forecast,test.values)['mape'])
        y_test.append(test)
        y_pred.append(future_forecast)
        model_fit=stepwise_model.predict_in_sample()
        pred_train_list.append([model_fit])
        
        
    return time[start_size-1:],y_test,y_pred,accuracy,pred_train_list