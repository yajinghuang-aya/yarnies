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
import app_pick_color
import app_pop_colors
import awesome_streamlit as ast

PAGES = {"Popular Colors":app_pop_colors,
		 "Pick Your Own Colors": app_pick_color
}

def main():
    """Main function of the App"""
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)

if __name__ == "__main__":
    main()