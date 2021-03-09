'''
Owner: Mit Jothiravi (1002321258)
Description: Answers to Problem 2 answer sheet (MIE 304)

Example command line usage:
    $ python problem_2.py --q1 True
'''

import argparse
import os

from utils.load_data import load_csv
from utils.viz_data import stem_plot, histogram_plot, cumulative_frequency_plot, box_plot, time_series_plot

# Globals
Q2_DATA_FP = "../data/02_data.csv"
DATA = load_csv(Q2_DATA_FP)

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--q1', help='Run question 1')
parser.add_argument('--q2', help='Run question 2')
parser.add_argument('--q3', help='Run question 3')
parser.add_argument('--q4', help='Run question 4')
parser.add_argument('--q5', help='Run question 5')
args = parser.parse_args()

if args.q1:
    stem_plot(DATA)

if args.q2:
    bins = 6
    histogram_plot(DATA, bins)

if args.q3:
    bins = 6
    cumulative_frequency_plot(DATA, bins)

if args.q5:
    time_series_plot(DATA)
