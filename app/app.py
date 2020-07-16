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


st.title("""
Dream in *Colors* 
""")

st.text("Color Trends Forecast for Yarns")





#st.info("Upload an image of yarn and see how popular its color has been and will be!")


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


st.plotly_chart(get_plotly_H_6(data))
st.plotly_chart(get_plotly_SxL_pop(data,hue=4))

#st.write("drop here")
#image_file = st.file_uploader("""Upload your own yarn color image here!""",type=['jpg','png','jpeg'])
#if image_file is not None:#
#	our_image = Image.open(image_file)
#	st.text("Image uploaded:")
	# st.write(type(our_image))
#	st.image(our_image,width=350)

#else:
#	our_image = Image.open("test_image.png")
#	st.text("Test image:")
#	st.image(our_image,width=350)
#new_img = np.array(our_image.convert('RGB'))


#color_order,_,_=dominant_color(new_img)

#fig = dominant_color_plot(new_img)

#plt.rcParams["figure.figsize"] = (3,3)

#st.plotly_chart(fig)

#st.markdown("### Enter the rank of the color category you want information of :")

#number = st.number_input(' ', min_value=1, max_value=8,value=1,step=1)
#if number > 8 or number <1:
#    st.error("Please enter a number within [1,8]")

#h,s,l = oneD_to_3D_index(color_order[number-1])

#st.write('The color category you chose is : '+ hue_name[h]+', satuation '+str(satuation[s])+', lightness '+str(lightness[l]))

#st.write('\n')

#st.subheader("Color Popularity Trends ")
#st.plotly_chart(get_plotly_H('h'+str(h),data))
st.markdown("### Pick your own color and find out its Popularity!")

color = st.beta_color_picker('Pick A Color', '#2b82b5')
color_rgb=hex2rgb(color)
color_hsl=rgb2hsl(color_rgb)
pix_h,pix_s,pix_l=hsl2bucket(color_hsl)
#print(lightness[pix_l])
color_bucket='Hue = '+hue_name[pix_h]+', Saturation = '+str(satuation[pix_s])+', Lightness = '+str(lightness[pix_l])
st.info('The current color belongs to HSL bucket: '+color_bucket )


st.plotly_chart(get_plotly_H('h'+str(pix_h),data))

st.plotly_chart(get_plotly_SxL('s'+str(pix_s)+'l'+str(pix_l),data))






