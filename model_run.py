import numpy as ny  
import pandas as pd 
from model import *
import matplotlib.pyplot as plt
rank=pd.read_csv("color_rank_136.csv",header=None)

c=str(rank[0][9])

(X_train,y_train,X_valid,y_valid) = prepare_data(color_code=c)

model=random_forest(X_train,y_train)

preds_test,r2=model_predection(model,X_valid,y_valid)

accuracy=forecast_accuracy(preds_test, y_valid)
print(accuracy)


preds_train = model.predict(X_train)
#r2 = r2_score(y_train, preds)

plt.plot(range(77),y_train,color='orange')
plt.plot(range(77),preds_train,color='g')
plt.plot(range(80,97),y_valid,color='orange')
plt.plot(range(80,97),preds_test,color='g')
plt.show()