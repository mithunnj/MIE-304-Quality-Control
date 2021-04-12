'''
Owner: Mit Jothiravi (1002321258)
Description: Answers to Problem 7 answer sheet (MIE 304)

Example command line usage:
    $ python problem_7.py --q1 True
'''

import argparse
import os
from utils.load_data import load_csv

import pandas as pd
import numpy as np
import scipy.stats as stat
import matplotlib  
matplotlib.use('TkAgg') # This is for Debian OS issues
import matplotlib.pyplot as plt
import math

# Globals
Q6_DATA_FP = "../data/07_data.csv"
DATA = load_csv(Q6_DATA_FP)
P_VALUE_ALPHA = 0.05 # If p-value < alpha, reject null hypothesis

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--q1', help='Run question 1')
args = parser.parse_args()

if args.q1:

    # Step 1: Conduct Anderson-Darling Normality test
    result = stat.anderson(DATA['Layer thickness'], 'norm')

    AD = result.statistic # Parse the AD test value

    # Calculate P-value of Null Hypothesis: Data follows a Normal Distribution
    if AD >= .6:
        p = math.exp(1.2937 - 5.709*AD - .0186*(AD**2))
    elif AD >=.34:
        p = math.exp(.9177 - 4.279*AD - 1.38*(AD**2))
    elif AD >.2:
        p = 1 - math.exp(-8.318 + 42.796*AD - 59.938*(AD**2))
    else:
        p = 1 - math.exp(-13.436 + 101.14*AD - 223.73*(AD**2))

    # Reject/accept Null Hypthosis
    if (p < P_VALUE_ALPHA):
        print("The p-value: {}, should be REJECTED because it is less than Alpha: {}".format(p, P_VALUE_ALPHA))
    else:
        print("The p-value: {}, should be ACCEPTED because it is less than Alpha: {}".format(p, P_VALUE_ALPHA))

    # Print Anderson-Darling Normality test results with plot
    print("Anderson-Darling normality test result: {}".format(AD))

    


    
