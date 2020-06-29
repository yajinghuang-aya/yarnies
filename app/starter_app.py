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

st.write("""
# Dream in Colors 
Hello *world!*
""")


@st.cache
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


data = load_data()

filename = st.file_picker("Pick a file", folder="my_folder", type=("png", "jpg")) 
file_bytes = st.file_uploader("Upload a file", type=("png", "jpg"))

st.plotly_chart(get_plotly_H('h0',data))
st.plotly_chart(get_plotly_SxL('s0l2',data))






