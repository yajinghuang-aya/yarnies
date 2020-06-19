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

st.write("""
# Yarn stuff
Hello *world!*
""")

image = Image.open('../color_feature_hsl/dat_fit_h3.png')

st.image(image, caption='model ',  use_column_width=True)


columns=[]
columns.extend(['h'+str(i) for i in range(6)])
columns.extend(['s'+str(i) for i in range(5)])
columns.extend(['l'+str(i) for i in range(5)])

model_fit={}
#for i in columns:
i='h0'

@st.cache
def load_model_fit(i):
	fit=pd.read_csv("../color_feature_hsl/model_fit_"+i+".csv")
	return fit


#for i in columns:
#	model_fit[i]=load_model_fit(i)

#fig, ax = plt.subplots(figsize=(8,5))

#ax.plot(pd.to_datetime(model_fit[i]['time']),model_fit[i]['data']*100,color='orange',label='training data')
#ax.plot(pd.to_datetime(time[:divide]),preds_train*100,color='g',label='model')
#ax.plot(pd.to_datetime(time[83:100]),y_valid*100,color='orange',ls=":",label='test data')
#ax.plot(pd.to_datetime(time[83:100]),preds_test*100,color='g')
#ax.set_title("Category "+i.upper())
#ax.set_xlabel('year')
#ax.set_ylabel('% popularity')
#ax.legend(loc="upper left")
#years = mdates.YearLocator()
#ax.xaxis.set_major_locator(years)

#st.plotly_chart(fig)

#st.line_chart(model_fit[i])


#dat_rf=pd.read_csv("../color_feature_hsl/color_weekly_feature_to_month_h0.csv")
#time=dat_rf['month']


dat_marginal2D=(pd.read_csv("../pop_color_hsl_marginal2D_weekly.csv"))


from matplotlib import gridspec

hue_value=[0,60,120,180,240,300]
satuation_value=[10,30,50,70,90]
lightness_value=[10,30,50,70,90]


fig1= plt.figure(figsize=(13, 6)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1]) 
ax1 = plt.subplot(gs[1], polar=True)


#ax0 = plt.subplot(1, 2, 1, polar=True)


for h in hue_value:
    ax1.bar([(h)*np.pi/180], [1],
           color=hsl2rgb([(h+10)/360,0.1,0.5]),width=0.9,bottom=0.5)
ax1.set_yticks([])
ax1.set_xticks([])
ax1.set_title('palette')
i="s0l2"

ax0 = plt.subplot(gs[0])
#ax = plt.subplot(1, 2, 2,size=(8,5))
ax0.plot(pd.to_datetime(dat_marginal2D['time']),dat_marginal2D[i]/dat_marginal2D['tot'],
            color=hsl2rgb([200/360.,satuation_value[int(i[1])]/100.
                           ,lightness_value[int(i[3])]/100]))

datemin = np.datetime64('2016', 'Y')
datemax = np.datetime64('2021', 'Y') #+ np.timedelta64(1, 'Y')
ax0.set_xlim(datemin, datemax)

st.pyplot(fig1)



def rgb2hex(rgb):
    
    print(rgb)
    rgb=tuple(rgb)
    return '#'+('%02x%02x%02x' % rgb)

def rgb2hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0],rgb[1],rgb[2])

hexcode=[]
for h in hue_value:
    color=np.asarray((hsl2rgb([(h+10)/360,0.1,0.5])))
    color=color*255
    color=color.astype('int')
    print(color)
    hexcode.append(rgb2hex(color))

print(hexcode)
def get_plotly_subplots():

	fig = make_subplots(
	    rows=1,
	    cols=2,
	    subplot_titles=("trends", "palette"),
	    specs=
	        [[{'type': 'xy'}, {'type': 'polar'}]],column_widths=[0.8, 0.2]
	)

	fig.add_trace(go.Barpolar(r=[2]*6,
		theta=hue_value,
		width=[60]*6,
		marker_color=hexcode)
		
		, row=1, col=2)

	fig.update_layout(
		template=None,
	    polar = dict(
	      radialaxis = dict(showticklabels=False, ticks='',showgrid=False,showline=False),
	      angularaxis = dict(showticklabels=False, ticks='',showgrid=False,showline=False)))
	        #direction = "clockwise",
	        #period = 6))
	   
	fig.add_trace(go.Scatter(x=pd.to_datetime(dat_marginal2D['time']),y=dat_marginal2D[i]/dat_marginal2D['tot']), row=1, col=1)
	fig.update_layout(height=550, width=1000)
	return fig

st.plotly_chart(get_plotly_subplots())

def prepare_data(color_code='h0'):

	dat_rf=st.cache(pd.read_csv("color_feature_hsl/color_weekly_feature_to_month_"+color_code+".csv"))
	lenth=dat_rf.shape[0]
	#print(lenth)
	x=dat_rf.drop(columns=['Unnamed: 0','index','y']).copy()
	y=dat_rf['y'].copy()
	x['year']=x['month'].apply(lambda x: int(x[:4]))
	x['m']=x['month'].apply(lambda x: int(x[-2:]))
	x=x.drop(columns='month')

	divide=(int(lenth*0.8))
	X_train=x.iloc[:divide]
	y_train=y.iloc[:divide]
	X_valid=x.iloc[divide+3:]
	y_valid=y.iloc[divide+3:]


	return  (X_train,y_train,X_valid,y_valid)



#st.line_chart(chart_data)
