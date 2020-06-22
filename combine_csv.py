import pandas as pd

# combine csv files from parallel running
file_nyc='image_pop_color_hsl/pop_color_'
fav_nyc=pd.read_csv(file_nyc+'0.csv',header=0)
for i in range(1,200):
    print(i)
    fn_nyc=file_nyc+str(i)+'.csv'
    fnyc=pd.read_csv(fn_nyc,header=0)
    fav_nyc=fav_nyc.append(fnyc)
fav_nyc.to_csv('project_pop_color_hsl_200.csv')
