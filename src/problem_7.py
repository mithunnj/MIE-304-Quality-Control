'''
Owner: Mit Jothiravi (1002321258)
Description: Answers to Problem 7 answer sheet (MIE 304)

Example command line usage:
    $ python problem_7.py --q1 True
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
Q6_DATA_FP = "../data/07_data.csv"
DATA = load_csv(Q6_DATA_FP)
P_VALUE_ALPHA = 0.05 # If p-value < alpha, reject null hypothesis

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--q1', help='Run question 1')
parser.add_argument('--q2', help='Run question 2')
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
    stat.probplot(DATA['Layer thickness'], plot = plt)
    plt.show()

if args.q2:
    # Step 1: Convert data into natural logarithm
    ln_conv = np.log(DATA['Layer thickness'])

    # Step 2: Conduct Anderson-Darling Normality test
    result = stat.anderson(ln_conv, 'norm')
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
    print("\n(Q2): Anderson-Darling Normality Test results: ")
    if (p < P_VALUE_ALPHA):
        print("The p-value: {}, should be REJECTED because it is less than Alpha: {}".format(p, P_VALUE_ALPHA))
    else:
        print("The p-value: {}, should be ACCEPTED because it is less than Alpha: {}".format(p, P_VALUE_ALPHA))

    # Print Anderson-Darling Normality test results with plot
    print("Anderson-Darling normality test result: {}".format(AD))
    stat.probplot(DATA['Layer thickness'], plot = plt)
    #plt.show() # UNCOMMENT If you want to see the plot

    # Step 3: Construct Individuals Control Chart (I-Chart)
    ## All the variables that are computed here are from this document: https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Individuals_and_Moving_Range_Charts.pdf
    x_bar = sum(ln_conv)/len(ln_conv)
    R = list()
    for i in range(1, len(ln_conv)):
        range_calc = abs(ln_conv[i] - ln_conv[i-1])
        R.append(range_calc)

    R_bar = sum(R)/(len(ln_conv)-1)
    d2 = 1.128 # For k = 2, as outlined here https://andrewmilivojevich.com/d2-values-for-the-distribution-of-the-average-range/#:~:text=The%20columns%20and%20rows%20represent,the%20value%20of%20d2%3D1.128.
    sigma_hat = R_bar/d2
    m = 3 # This multiplier value is the usual default value

    ## Build the Individuals Control Chart
    print("\nIndividuals Chart (I-Chart) results: ")
    I_LCL = x_bar - 3*sigma_hat
    I_UCL = x_bar + 3*sigma_hat
    print("(Q5, Q6): Mean (x_bar AKA CL) of the I-Chart process is: {}".format(x_bar))
    print("(Q5): Control limits for I-Chart: LCL = {}, UCL = {}".format(I_LCL, I_UCL))

    # Step 4: Construct the Moving Range Control Chart limits
    print("\nMoving Range Chart (MR-Chart) results: ")
    MR_samples=[]
    MR_accum=0
    for i in range(0,len(ln_conv)-1):
        MR=np.abs(ln_conv[i+1]-ln_conv[i])
        MR_samples.append(MR)
        MR_accum=MR_accum +MR

    # NOTE: Ported over from Tutorial Code
    MR_bar=MR_accum/(len(ln_conv)-1)
    LCL_MR=0
    UCL_MR=3.267* MR_bar
    MR_sigma = MR_bar/d2

    print("(Q3): MR-Bar: {}".format(MR_bar))
    print("(Q4): MR-Std: {}".format(MR_sigma))
    print("MR Control limits: LCL = {}, UCL = {}".format(LCL_MR, UCL_MR))

    ### Remaining questins (7-9)
    # Question 7
    n=1
    mu_1=2.7
    mu_0=x_bar
    Delta=mu_1-mu_0
    k=Delta/MR_sigma
    beta=stat.norm.cdf(3-k*np.sqrt(n),0,1)-stat.norm.cdf(-3-k*np.sqrt(n),0,1)

    print('\n(Q7): beta=', beta)
    print('prob of detecting shift on first sample=', 1-beta)

    # Question 8
    ARL=1/(1-beta)
    print('\n(Q8): ARL=', ARL)





    


    
