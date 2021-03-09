'''
Owner: Mit Jothiravi 1002321258
Description: Utils to load a .csv file
'''
import os
import sys
import pandas as pd

def load_csv(fp):
    '''
    Input: fp <str> - The filepath to the .csv data file
    
    Take in a .csv filepath, and convert it to a parseable pd dataframe.
    '''

    # Check that filepath exists
    if not os.path.isfile(fp):
        sys.exit('ERROR: <load_csv> assertion failed to find filepath: {}'.format(fp))

    # Check if file is a .csv
    if not fp.endswith('.csv'):
        sys.exit('ERROR: <load_csv> assertion failed to find .csv file : {}'.format(fp))

    # Load, convert .csv into Panda datastructure
    data = pd.read_csv(fp)

    return data