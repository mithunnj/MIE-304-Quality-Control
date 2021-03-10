'''
Owner: Mit Jothiravi (1002321258)
Description: Answers to Problem 5 answer sheet (MIE 304)

Example command line usage:
    $ python problem_5.py --q1 True
'''

import argparse
import os

from utils.load_data import load_csv
from utils.calc_stats import r_chart_values

# Globals
Q5_DATA_FP = "../data/05_data.csv"
DATA = load_csv(Q5_DATA_FP)

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--q1', help='Run question 1')
parser.add_argument('--q2', help='Run question 2')
parser.add_argument('--q3', help='Run question 3')
parser.add_argument('--q4', help='Run question 4')
args = parser.parse_args()

if args.q1:
    sample_size = 4 # In each sample, there are 4 values 

    # Calculate R-chart values
    R_VALS, R_BAR, R_UCL, R_LCL = r_chart_values(DATA, sample_size)
    
    # Calculate X-bar chart values

    print(R_BAR)
    print(R_UCL)
    print(R_LCL)

 
    
