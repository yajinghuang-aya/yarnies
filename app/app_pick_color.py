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
from app_util import *

def write():
	@st.cache(allow_output_mutation=True)
	def load_data():
		data=pd.read_csv("data_final_hsl_cat_weekly.csv")
		data.drop(columns='Unnamed: 0',inplace=True)

		return data

	hue_value=[0,60,120,180,240,300]
	satuation_value=[10,30,50,70,90]
	lightness_value=[10,30,50,70,90]
	hue=[[330,30],[30,90],[90,150],[150,210],[210,270],[270,330]]
	satuation=[[0,20],[20,40],[40,60],[60,80],[80,100]]
	lightness=[[0,20],[20,40],[40,60],[60,80],[80,100]]
	hue_name=['Red','Yellow','Green','Cyan','Blue','Magenta']

	data = load_data()


	color = st.beta_color_picker('Pick A Color', '#2b82b5')
	color_rgb=hex2rgb(color)
	color_hsl=rgb2hsl(color_rgb)
	pix_h,pix_s,pix_l=hsl2bucket(color_hsl)
	#print(lightness[pix_l])
	color_bucket='Hue = '+hue_name[pix_h]+', Saturation = '+str(satuation[pix_s])+', Lightness = '+str(lightness[pix_l])
	st.write('The current color belongs to HSL bucket: '+color_bucket )


	st.plotly_chart(get_plotly_H('h'+str(pix_h),data))

	st.plotly_chart(get_plotly_SxL('s'+str(pix_s)+'l'+str(pix_l),data))
