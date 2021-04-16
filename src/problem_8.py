'''
Owner: Mit Jothiravi (1002321258)
Description: Answers to Problem 8 answer sheet (MIE 304)

Example command line usage:
    $ python problem_8.py --q1 True
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
Q8_DATA_FP = "../data/08_data.csv"
DATA = load_csv(Q8_DATA_FP)

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--q1', help='Run question 1')
parser.add_argument('--q2', help='Run question 2')
parser.add_argument('--q3', help='Run question 3')
parser.add_argument('--q4', help='Run question 4')
parser.add_argument('--q6', help='Run question 6')
parser.add_argument('--q7', help='Run question 7')
args = parser.parse_args()

if args.q1:
    '''
    Will address Q1 Problem 8 sheet

    Creating a Fraction Non-Conforming Control chart and checking if the process is 
    in control.
    '''

    # Define sample size (n) and the number of samples (m) given 
    ## from the problem
    ## NOTE: More info on building fraction nonconforming control charts can be found here: https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc332.htm
    n = 50 # 50 claims per day -> sample size
    m = 20 # 20 days in the data -> # of samples

    CL = DATA['insurance_nonc'].sum() / (m*n) # p_bar 
    UCL = CL + 3* np.sqrt((CL*(1-CL))/n) 
    LCL  = max(0, CL - 3* np.sqrt((CL*(1-CL))/n) ) # If the value is negative, we set it to 0.

    print("\nControl limits for Fraction Non-Conforming control chart:\n- CL: {}\n- UCL: {}\n- LCL: {}".format(CL, UCL, LCL))

    p_hat = DATA['insurance_nonc']/n # The fraction non conforming

    # Check for data points that are out of statistical control
    print("\nChecking if process is out of statistical control for Fraction Non-Conforming Control Chart.")
    for i in range (1, m+1):
        if (p_hat[i-1]> UCL or p_hat[i-1]< LCL):
            print("Data point: {}, Value: {}, is out of control - Process is OUT OF CONTROL".format(i, p_hat[i-1]))

if args.q2:
    '''
    Will address Q2 Problem 8 sheet

    Adjusting the control limits of the Fraction Non-Conforming Control Chart by 
    removing the out of control data point.
    '''
    n = 50 # 50 claims per day -> sample size
    m = 20 # 20 days in the data -> # of samples
    problem_index = 15 # From the last question, the problem index was at index 15, data array position 16. So we remove that one.

    CL = (DATA['insurance_nonc'].sum()-DATA['insurance_nonc'][15])/(n*(m-1)) # m-1 because we removed the sum of one point
    UCL = CL + 3* np.sqrt((CL*(1-CL))/n)
    LCL  = max(0, CL - 3* np.sqrt((CL*(1-CL))/n) )
    print("\nControl limits for ADJUSTED Fraction Non-Conforming control chart:\n- CL: {}\n- UCL: {}\n- LCL: {}".format(CL, UCL, LCL))

    ''' 
    #NOTE: I copied over this code where they check if the process is still in control. You can ignore for now.
    for i in range (1, m+1):
        if(p_hat[i-1]> UCL or p_hat[i-1]< LCL):
            print(i, 'is out of control')
    # 16 remains OOC, no new point OOC, so these are the final values for control limits
    '''

if args.q3:
    '''
    Will address Q3 Problem 8 sheet

    Creating NP-Chart control limits from the Control Limits computed from the Fraction Non-Conforming Control chart
    by just multiplying by n (sample size)
    '''
    #NOTE: Depending on the control limit values you want, change the CL, UCL, LCL code that you copied. 

    # Calculating the control charts copied over from Q1
    n = 50 # 50 claims per day -> sample size
    m = 20 # 20 days in the data -> # of samples

    CL = DATA['insurance_nonc'].sum() / (m*n) # p_bar 
    UCL = CL + 3* np.sqrt((CL*(1-CL))/n) 
    LCL  = max(0, CL - 3* np.sqrt((CL*(1-CL))/n) ) # If the value is negative, we set it to 0.

    # Create control limits for NP Chart
    CL *= n
    UCL *= n 
    LCL *= n

    print("\nControl limits for NP control chart:\n- CL: {}\n- UCL: {}\n- LCL: {}".format(CL, UCL, LCL))

if args.q4:
    '''
    Will address Q4, Q5 Problem 8 Sheet

    This aimed to calculate Variable-Width Control Charts. 
    '''
    # They took the ratio between the two columns that they were analyzing
    p_hat = DATA['num_late'].sum() / DATA['num_app'].sum() # p_hat for Q4
    CL = p_hat

    print("The CL for the Variable-Width Control Limit is: {}".format(CL))


    ## Below is checking of the process is out of control using the 
    ### Variable-Width Control Chart
    ### NOTE: Not sure why they did it this way, but you can use this
    m=20
    DATA['UCL']=CL+3*np.sqrt((CL*(1-CL))/DATA['num_app'])
    DATA['LCL']=CL-3*np.sqrt((CL*(1-CL))/DATA['num_app'])
    DATA.loc[DATA['LCL']<0,'LCL']=0

    p_hat=DATA['num_late']/DATA['num_app'] # p_hat for Q5

    for i in range (1, m+1):
        if(p_hat[i-1]> DATA.UCL[i-1] or p_hat[i-1]< DATA.LCL[i-1]):
            print(i, 'is out of control')

if args.q6:
    '''
    Will address Q6 Problem 8 Sheet

    This is like before, once you determine a point is out of control, they want us to remove
    it, and then recompute the control limits.
    '''
    #Question #6
    index_remove = 14 # This was the index of the out of bounds point to remove from the Control Limit lines
    CL=(DATA['num_late'].sum()-DATA['num_late'][index_remove])/(DATA['num_app'].sum()-DATA['num_app'][index_remove])
    print(CL)

    DATA['UCL']=CL+3*np.sqrt((CL*(1-CL))/DATA['num_app'])
    DATA['LCL']=CL-3*np.sqrt((CL*(1-CL))/DATA['num_app'])
    DATA.loc[DATA['LCL']<0,'LCL']=0

    p_hat=DATA['num_late']/DATA['num_app']

    for i in range (1, m+1):
        if(p_hat[i-1]> DATA.UCL[i-1] or p_hat[i-1]< DATA.LCL[i-1]):
            print(i, 'is out of control')

    ## And then they recheck the control limit information
    CL=(DATA['num_late'].sum()-DATA['num_late'][14]-DATA['num_late'][11])/(DATA['num_app'].sum()-DATA['num_app'][14]-DATA['num_app'][11])
    print(CL)

    DATA['UCL']=CL+3*np.sqrt((CL*(1-CL))/DATA['num_app'])
    DATA['LCL']=CL-3*np.sqrt((CL*(1-CL))/DATA['num_app'])
    DATA.loc[DATA['LCL']<0,'LCL']=0

    p_hat=DATA['num_late']/DATA['num_app']

    for i in range (1, m+1):
        if(p_hat[i-1]> DATA.UCL[i-1] or p_hat[i-1]< DATA.LCL[i-1]):
            print(i, 'is out of control')

if args.q7:
    '''
    Will address Q7 Problem 8 Sheet

    Simple average sample size calculator
    '''
    n_bar = math.ceil(DATA['num_app'].mean()) # Must round up 
    print(n_bar)

## NOTE: For Q8, Q9, Q10 check the Tutorial 08 Questions.ipynb notebook here: https://colab.research.google.com/drive/1ZKrslZTxo6th6Jc42w_L41M1zZFzMopa#scrollTo=eXWbWXkZggei
            




    


    
