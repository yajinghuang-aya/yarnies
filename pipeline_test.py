#pipeline test
import pandas as pd
from color_time_util import *

import pickle

#sample ds pat vs month
data=pd.read_csv("pat_time_count_nyc_sample.csv",header=0,sep="^")
sample=get_df_sample(data,frac=0.01)

#clustering color
df_pattern_to_color(sample)

# color dict per month
color_dict_months=color_vs_month(sample)
print(type(color_dict_months))
print(color_dict_months.keys())
#json.dump(color_dict_months, open( "color_vs_month_sample.json", 'w' ) )

output = open('color_vs_month_sample.pkl', 'wb')
pickle.dump(color_dict_months, output)
output.close()

