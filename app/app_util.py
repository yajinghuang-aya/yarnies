import streamlit as st
import pandas as pd
#from ../model import random_forest,model_predection,forecast_accuracy
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from colour import hsl2rgb
import matplotlib.dates as mdates
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE 
import cv2
from colour import Color

hue_value=[0,60,120,180,240,300]
satuation_value=[10,30,50,70,90]
lightness_value=[10,30,50,70,90]
hue_name=['Red','Yellow','Green','Cyan','Blue','Magenta']
hue=[[330,30],[30,90],[90,150],[150,210],[210,270],[270,330]]
satuation=[[0,20],[20,40],[40,60],[60,80],[80,100]]
lightness=[[0,20],[20,40],[40,60],[60,80],[80,100]]

def rgb2hex(rgb):
    
    #print(rgb)
    rgb=tuple(rgb)
    return '#'+('%02x%02x%02x' % rgb)

def rgb2hsl(pix):
    pix=np.asarray(pix)/255.
    pixcolor=Color(rgb=pix)
    hsl=pixcolor.hsl
    color_hsl=np.zeros(3)
    color_hsl[0]=hsl[0]*360
    color_hsl[1]=hsl[1]*100
    color_hsl[2]=hsl[2]*100
    return color_hsl

def oneD_to_3D_index(o):

    i1=o//25
    i2=(o%25)//5
    i3=(o%25)%5
    return i1,i2,i3

def rolling_rf_data_pre(data,col='s0l2'):  #data weekly

    #data=df.copy()
    for j in range(1,len(data.columns)):
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


def forecast_rf(df_forecasting,last_feature):
    
    
    x=df_forecasting.iloc[:,1:].copy()
    #print(x.shape)
    y=df_forecasting.iloc[:,0].copy()
    
    mdl=RandomForestRegressor(n_estimators=15,random_state=0,min_samples_leaf=1)

    mdl.fit(x,y)
    #print('y',y)
    model_fit= mdl.predict(x)
    #print('m',model_fit)
    x_current=np.asarray(last_feature).reshape(1, -1) 
    forecast= mdl.predict(x_current)
    return df_forecasting.index,model_fit,forecast



def forescast_feature_pre(data,df_forecasting,col='s0l2'):
    for j in range(1,len(data.columns)-1):
        data.iloc[:,j]=data.iloc[:,j]/data.iloc[:,-1]
    pop=(data[col].copy()).rolling(6).mean()
    #print(pop)
    feature=list(pop[-8:])
    lag_t=df_forecasting.columns[9:]
    #t-12
    
    pop_month=df_forecasting[col]
    
    for t in lag_t:
        t_index=-int(t[2:])
        feature.append(pop_month[t_index])
    return feature
    
    
@st.cache(allow_output_mutation=True)    
def get_plotly_SxL(col_test,data):
    
    s=int(col_test[1])
    l=int(col_test[-1])

    df_forecasting=rolling_rf_data_pre(data,col=col_test)
    f=forescast_feature_pre(data,df_forecasting,col=col_test)
    t,model,pred=forecast_rf(df_forecasting,f)


    #fig = go.Figure()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(" ", "Colors in the Palette"),
        specs=
            [[{'type': 'xy'}, {'type': 'polar'}]],column_widths=[0.75, 0.25]
    )


    time=list(t)
    time.append(pd.to_datetime("2020-06-30"))
    forecast=list(100*model)
    forecast.append(pred[0]*100)

    fig.add_trace(go.Scatter(x=time,y=forecast,
                             mode='lines', name='model & forecast',line=dict(color="#80bf40",dash='dot')),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=100*df_forecasting[col_test],
                              mode='lines', name='data',line=dict(color="#409fbf")),
                 row=1, col=1)


    title='Saturation = '+str(satuation[s]) + ', Lightness = '+str(satuation[l])

    fig.update_layout(
        title=title,
        title_x=0.,
        width=1050,
        height=650,
        xaxis_title="time",
        yaxis_title="% popularity",
        font=dict(
            family="Courier New, monospace",
            size=17,
            color="#7f7f7f"),
        xaxis=dict(range=[pd.to_datetime("2016-06-30"),pd.to_datetime("2020-06-30")],
                   rangeselector=dict(
                buttons=list([

                    dict(count=6,
                         label="last 6 months",
                         step="month",
                         stepmode="backward"),

                    dict(count=1,
                         label="last year",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
                  ),
            legend=dict(
            x=0,
            y=0.99,
            traceorder='normal',
            font=dict(
                size=12,),
        ),
    )

    hexcode=[]
    for h in hue_value:
        color=np.asarray((hsl2rgb([(h+10)/360,satuation_value[s]/100,lightness_value[l]/100])))
        color=color*255
        color=color.astype('int')
        #print(color)
        hexcode.append(rgb2hex(color))

    fig.add_trace(go.Barpolar(r=[2]*6,
            theta=hue_value,
            width=[60]*6,
            marker_color=hexcode,
            showlegend = False,
            hoverinfo='none'
     ), row=1, col=2)

    fig.update_layout(
            template=None,
            polar = dict(
              radialaxis = dict(showticklabels=False, ticks='',showgrid=False,showline=False),
              angularaxis = dict(showticklabels=False, ticks='',showgrid=False,showline=False,rotation=90),
           ),

            )

    return fig

@st.cache(allow_output_mutation=True)
def get_plotly_H(col_test,data):
    df_forecasting=rolling_rf_data_pre(data,col=col_test)
    f=forescast_feature_pre(data,df_forecasting,col=col_test)
    t,model,pred=forecast_rf(df_forecasting,f)

    time=list(t)
    time.append(pd.to_datetime("2020-06-30"))
    forecast=list(100*model)
    forecast.append(pred[0]*100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time,y=forecast,
                                 mode='lines', name='model & forecast',line=dict(color="#80bf40",dash='dot')))
    fig.add_trace(go.Scatter(x=t, y=100*df_forecasting[col_test],
                                  mode='lines', name='data',line=dict(color="#409fbf")))
    h=int(col_test[1])




    title='Hue = '+hue_name[h]
    fig.update_layout(
            title=title,
            width=850,
            height=550,
            xaxis_title="time",
            yaxis_title="% popularity",
            font=dict(
                family="Courier New, monospace",
                size=17,
                color="#7f7f7f"),
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            xaxis=dict(range=[pd.to_datetime("2016-06-30"),pd.to_datetime("2020-06-30")],
                       rangeselector=dict(
                    buttons=list([

                        dict(count=6,
                             label="last 6 months",
                             step="month",
                             stepmode="backward"),

                        dict(count=1,
                             label="last year",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
                      ),
                legend=dict(
                x=0,
                y=0.99,
                traceorder='normal',
                font=dict(
                    size=12,),
            ),
                
                 #paper_bgcolor='rgba(0,0,0,0)')
                plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(showgrid=True,gridcolor="#DCDCDC")
    fig.update_xaxes(showgrid=True,gridcolor="#DCDCDC")
    return fig


def remove_background(image):
    index_del=[]
    for i in range(image.shape[0]):
        if image[i][0]>235 and image[i][1]>235 and image[i][2]>235:
            index_del.append(i)
        if image[i][0]<50 and image[i][1]<50 and image[i][2]<50:
            index_del.append(i)

    if index_del:
        image=np.delete(image,index_del,axis=0)
    return image


def prepare_image(image):   
    wid=int(image.shape[1]*0.6)
    height=int(image.shape[0]*0.6)

    image=cv2.resize(image,(wid,height), interpolation = cv2.INTER_AREA)
    wid=image.shape[1]
    height=image.shape[0]
    image=image[int(wid*0.2):int(wid*0.8),int(height*0.2):int(height*0.8)]
    #image=flip_rgb(image)
    #img = Image.fromarray(image,"RGB")
    #img.save('small2.jpg')

    image=image.reshape([image.shape[0]*image.shape[1],3])
    image=remove_background(image)

    return image

@st.cache(allow_output_mutation=True)
def dominant_color(image):

    hsl_cat=np.zeros((6,5,5))
    image=(prepare_image(image).astype('uint8'))
    red=0
    blue=0
    for i in range(image.shape[0]):
        pix_hsl=rgb2hsl(image[i])
        if image[i][0]>image[i][2]:
            red+=1
        else: blue+=1
        #print(image[i],pix_hsl)
        for i in range(1,6):
            if pix_hsl[0]>=hue[i][0] and pix_hsl[0]<hue[i][1]:
                pix_h=i
        if pix_hsl[0]>=330 or pix_hsl[0]<30:
                pix_h=0
        for i in range(5):
            if pix_hsl[1]>=satuation[i][0] and pix_hsl[1]<satuation[i][1]:
                pix_s=i
            if pix_hsl[2]>=lightness[i][0] and pix_hsl[2]<lightness[i][1]:
                pix_l=i
        hsl_cat[pix_h,pix_s,pix_l]+=1
        #print([pix_h,pix_s,pix_l])
    color_cat_flat=hsl_cat.reshape(150)
    frequency={}
    for i in range(150):
        frequency[i]=color_cat_flat[i]

    color_count=np.arange(150)
    for i in range(150):
        color_count[i]=frequency[i]

    color_order=np.argsort(color_count)[::-1]
    color_rank_rgb=[]
    for i in color_order:
        i1,i2,i3=oneD_to_3D_index(i)
        color_hsl=np.asarray([hue_value[i1]/360.,satuation_value[i2]/100.,lightness_value[i3]/100.])
        color_rgb=np.asarray(hsl2rgb(color_hsl))

        color_rank_rgb.append(color_rgb)

    return color_order,color_rank_rgb,color_count


@st.cache(allow_output_mutation=True)
def dominant_color_plot(image):

    color_order,color_rank_rgb,color_count=dominant_color(image)
    bar_x=[]
    bar_y=[]
    bar_x=np.arange(1,9)
    for i in range(8):
       bar_y.append(color_count[color_order[i]])




    hexcode=[]
    for i in range(8):
        color=np.asarray(color_rank_rgb[i])
        color=color*255
        color=color.astype('int')
        #print(color)
        hexcode.append(rgb2hex(color))

    fig = go.Figure(data=[go.Bar(
            x=bar_x, y=bar_y, marker_color=hexcode
            )])

    fig.update_layout(
    title="Dominant Color Categories",
    title_x=0.,
    width=550,
    height=450,
    xaxis=dict(title="color Category rank", tickmode='linear'),
    yaxis_title="number of pixels",
    font=dict(
        family="Courier New, monospace",
        size=17,
        color="#7f7f7f"))
   
    print(color_order)

    return fig
