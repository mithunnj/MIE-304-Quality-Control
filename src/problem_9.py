'''
Owner: Mit Jothiravi (1002321258)
Description: Answers to Problem 9 answer sheet (MIE 304)

Example command line usage:
    $ python problem_9.py --q1 True
'''

import argparse
import os
from utils.load_data import load_csv
import sys

import pandas as pd
import numpy as np
import scipy.stats as stat
import matplotlib  
matplotlib.use('TkAgg') # This is for Debian OS issues
import matplotlib.pyplot as plt
import math 

# Globals
DATA_FP = "../data/09_data.csv"
DATA = load_csv(DATA_FP)

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--q1', help='Run question 1')
parser.add_argument('--q3', help='Run question 3')
args = parser.parse_args()

if args.q1:
    '''
    Answer for Q1, Q2 for Sheet 9:

    It wasn't clear what kind of chart to construct, so I showed how to parse
    dataframe for a column and ignore the NANs
    '''
    # Fetch all the data from column: num_nonc that is not NAN
    parsed_data = DATA.dropna(how='all', subset=['num_nonc'])
    m = len(parsed_data)

    print(parsed_data)

if args.q3:
    '''
    Answer for Q3, Q4 for Sheet 9:

    Shows how to control a U-Chart and dtermine if the process is in control using a 
    varyling limit chart.
    '''
    m = len(DATA['num_cit'])
    CL = DATA['num_wo'].sum()/DATA['num_cit'].sum()

    # For this method, we have to store the control limits for each iteration 
    DATA['UCL']=CL+3*np.sqrt(CL/DATA['num_cit'])
    DATA['LCL']=CL-3*np.sqrt(CL/DATA['num_cit'])
    DATA.loc[DATA['LCL']<0, 'LCL']=0

    u_hat=DATA['num_wo']/DATA['num_cit']

    # Check if the process is in control
    for i in range(1,m+1):
        if (u_hat[i-1]>DATA.UCL[i-1] or u_hat[i-1]<DATA.LCL[i-1]):
            print(i,'is out of control')

    print("Process control check done.")

