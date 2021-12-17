# TODO: Imports
import pandas as pd
import matplotlib.pyplot as plt
#  Read csv dataset
data=pd.read_csv(r"D:\1Downloads\sscore.csv")
# TODO: Plotting
x=data["Hours"]
y=data["Scores"]
plt.plot(x,y, 'ro')
plt.show()
# TODO: Preprocessings
# TODO: Model Training
# TODO: Result analysis