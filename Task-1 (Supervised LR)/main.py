# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read csv dataset
data=pd.read_csv(r"D:\1Downloads\sscore.csv")

# Plotting
x=data["Hours"]
y=data["Scores"]

plt.plot(x,y, 'ro')
plt.show()

# Preprocessings
xarray=np.array(x)
yarray=np.array(y)

xarray=xarray.reshape(-1,1)
yarray=yarray.reshape(-1,1)

# Model Training
lrmodel=LinearRegression()
lrmodel.fit(xarray,yarray)

# Result analysis
score=lrmodel.score(xarray, yarray)*100
print(f"Score of the model is: {round(score,2)}")

predict=lrmodel.predict([[9.25]])
print(f"Predicted value of score for 9.25 hrs/day is: {round(predict[0][0])}")

slope=lrmodel.coef_[0][0]
yintercept=lrmodel.intercept_[0]

yline=yintercept+slope*(np.array(x))
plt.plot(x,y, 'ro')
plt.plot(x, yline)
plt.show()
