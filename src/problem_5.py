'''
Owner: Mit Jothiravi (1002321258)
Description: Answers to Problem 5 answer sheet (MIE 304)

Example command line usage:
    $ python problem_5.py --q1 True
'''

import argparse
import os

from utils.load_data import load_csv
from utils.viz_data import x_bar_r_chart_plot
from utils.calc_stats import r_chart_values, x_bar_chart_values, chart_control_type

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
    # NOTE: Custom implementation for future questions will require changes to the 
    ## parsing mechanism within r_chart_values function
    R_VALS, R_BAR, R_UCL, R_LCL = r_chart_values(DATA, sample_size)
    
    # Calculate X-bar chart values
    X_BAR_VALS, X_BAR_BAR, X_BAR_UCL, X_BAR_LCL = x_bar_chart_values(DATA, sample_size, R_BAR)

    # Check if R-chart & X-bar chart are in control
    chart_control_type('R', R_VALS, R_UCL, R_LCL)
    chart_control_type('X-bar', X_BAR_VALS, X_BAR_UCL, X_BAR_LCL)

    # Calculate total number of samples takes (m)
    ## NOTE: Change the two categories in this df was "Sample Number" and "Voltage". Change accordingly
    m = len(DATA["Sample Number"].unique().tolist())
 
    # Plot X-bar chart
    x_bar_r_chart_plot(m, X_BAR_VALS, X_BAR_BAR, X_BAR_LCL, X_BAR_UCL, 'X-bar')

    # Plot R chart
    x_bar_r_chart_plot(m, R_VALS, R_BAR, R_LCL, R_UCL, 'R')

    
