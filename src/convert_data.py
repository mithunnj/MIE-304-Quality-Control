'''
Project related code to convert the string data into official format
This is to be run locally, and the data filepath (DATA_FP) should point to where the .csv file is stored
'''
import pandas as pd
import os
import sys
import datetime
import matplotlib  
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import numpy as np

DATA_FP = os.path.dirname(os.getcwd()) + "/data/MiningProcess_Flotation_Plant_Database.csv"
NEW_DATA_FP = os.path.dirname(os.getcwd()) + "/data/MiningProcess_Flotation_Plant_Database_FLOATS.csv"

# Load data
df = pd.read_csv(NEW_DATA_FP)
column_names = list(df.columns.values)

print(column_names)
x_title = 'date'
y_title = "Ore Pulp Density"

x = df[x_title][::350].tolist()
y = df [y_title][::350].tolist()

dates = [pd.to_datetime(d.split(' ')[0]) for d in x]

plt.scatter(dates,y)
plt.title("Scatter Diagram Quality Tool")
plt.xlabel(x_title)
plt.ylabel(y_title)
plt.show()