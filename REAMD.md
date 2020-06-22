# Dream in colors
## code for predicting color trends in yarn industry

This project utilized color information from project images from [ravelry](https://www.ravelry.com/) to model color trends and predict color poppularity. 
Images were downloaded via ravelry api with time information. Clothing items were detected and cropped using faster r-cnn with inception restnet.
Pixel colors then were binned into 150 HSL categories. Color popularities of each category for each image along with its time info were aggreaged into
one single dataframe. The dataframe was grouped by weeks to obtain features and by months to obtain labels. Features and labels then were fed into
a random forest regression model. The model with lasted feature can predict color popluarity in the coming month. 

## file description:
project_load_txt.py:
returns project ID, time stamp and photo url from ravelry API


download_and_crop.py: download image from url and crop out clothing items

color_get_pop_image: bin pixel colors into HSL categories

project_color_popularity_data_processing_hsl.ipynb: explore color populariry and create dataframe for modeling

color_weekly_feature_hsl.py: prepare dataframe for modeling

model_run.py: fit random forest regression, return accuracy and plots

