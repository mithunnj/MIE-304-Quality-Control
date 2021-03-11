'''
Owner: Mit Jothiravi (1002321258)
Description: Answers to Problem 6 answer sheet (MIE 304)

Example command line usage:
    $ python problem_6.py --q1 True
'''

import argparse
import os

from utils.load_data import load_csv
from utils.viz_data import x_bar_any_chart_plot
from utils.calc_stats import s_chart_values, x_bar_s_chart_values, chart_control_type

# Globals
Q6_DATA_FP = "../data/06_data.csv"
DATA = load_csv(Q6_DATA_FP)

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--q5', help='Run question 5')
args = parser.parse_args()

if args.q5:
    sample_size = 3 # In each sample, there are 3 values 

    # Calculate S-chart values
    # NOTE: Custom implementation for future questions will require changes to the category information to parse dataframe within the function
    S_VALS, S_BAR, S_UCL, S_LCL = s_chart_values(DATA, sample_size, 'Thickness')

    # Calculate X-bar chart values
    X_BAR_VALS, X_BAR_BAR, X_BAR_UCL, X_BAR_LCL = x_bar_s_chart_values(DATA, sample_size, S_BAR, 'Thickness')

    # Check if R-chart & X-bar chart are in control
    chart_control_type('S', S_VALS, S_UCL, S_LCL)
    chart_control_type('X-bar', X_BAR_VALS, X_BAR_UCL, X_BAR_LCL)

    # Calculate total number of samples takes (m)
    m = len(DATA["Sample Number"].unique().tolist())
 
    # Plot X-bar chart
    x_bar_any_chart_plot(m, X_BAR_VALS, X_BAR_BAR, X_BAR_LCL, X_BAR_UCL, 'X-bar')

    # Plot S chart
    x_bar_any_chart_plot(m, S_VALS, S_BAR, S_LCL, S_UCL, 'S')

    
