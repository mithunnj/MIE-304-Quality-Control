'''
Owner: Mit Jothiravi (1002321258)
Description: Answers to Problem 2 answer sheet (MIE 304)

Example command line usage:
    $ python problem_2.py --q1 True
'''

import argparse
import os

from utils.load_data import load_csv
from utils.viz_data import viz_data

# Globals
Q2_DATA_FP = "../data/02_data.csv"
DATA = load_csv(Q2_DATA_FP)

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--q1', help='Run question 1')
parser.add_argument('--q2', help='Run question 2')
parser.add_argument('--q3', help='Run question 3')
args = parser.parse_args()

if args.q1:
    viz_data(DATA, stem_plot=True)

if args.q2:
    viz_data(DATA, histogram=True, bins=6)

if args.q3:
    viz_data(DATA, cumulative_frequency=True, bins=6)
