'''
Project related code to convert the string data into official format
This is to be run locally, and the data filepath (DATA_FP) should point to where the .csv file is stored
'''
import pandas as pd
import os
import sys
import datetime

DATA_FP = os.path.dirname(os.getcwd()) + "/data/MiningProcess_Flotation_Plant_Database.csv"
NEW_DATA_FP = os.path.dirname(os.getcwd()) + "/data/MiningProcess_Flotation_Plant_Database_FLOATS.csv"

# Load data
df = pd.read_csv(DATA_FP)
column_names = list(df.columns.values)

# Fix all data except for timestamp data
edit_columns = [x for x in column_names if x != 'date'] # Get column names in dataframe and remove timestamp column

for column in edit_columns: 
    edit_data = list() # Will store the converted data

    # Convert all commas in data in this column to '.' to convert to float
    for data in df[column]:
        edit_data.append(float(data.replace(',','.')))

    df[column] = edit_data # Overwrite the column data with the correct format

# Save the converted data to a new .csv file
df.to_csv(NEW_DATA_FP)

