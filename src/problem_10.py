'''
Owner: Mit Jothiravi (1002321258)
Description: Answers to Problem 10 answer sheet (MIE 304)

Example command line usage:
    $ python problem_10.py --q1 True
'''

import argparse
import os
from utils.load_data import load_csv
from utils.calc_stats import A2, D3, D4
import sys

import pandas as pd
import numpy as np
import scipy.stats as stat
import matplotlib  
matplotlib.use('TkAgg') # This is for Debian OS issues
import matplotlib.pyplot as plt
import math 

# Globals
DATA_FP = "../data/10_data.csv"
DATA = load_csv(DATA_FP)

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--q1', help='Run question 1')
args = parser.parse_args()

if args.q1:
    '''
    This question addresses Q1, Q2
    '''

    # Step #1: Build X-Bar control limits
    x_bars = list()
    r_vals = list()
    sample_numbers = list(DATA.Sample.unique()) # Get a list of unique elements from the Sample Number column

    ## Construct list of x_bars (calculate the sample means)
    for i in sample_numbers:
        sample_data = DATA.loc[DATA['Sample'] == i, "Vol"]

        # Calculate R and X_bar values and store
        x_bars.append(sample_data.mean())
        r_vals.append(sample_data.max() - sample_data.min())

    ## Calculate the CL information for both X-Bar and R charts
    x_bar_bar = x_bars.sum() / len(x_bars)
    r_bar = r_vals.sum() / len(r_vals)

    ## Calulate the Control Limit lines for X-Bar and R charts
    X_bar_UCL = x_bar_bar + A2[4]*r_bar
    X_bar_LCL = x_bar_bar - A2[4]*r_bar

    R_bar_LCL = D3[4]*r_bar
    R_bar_UCL = D4[4]*r_bar

    