'''
Owner: Mit Jothiravi (1002321258)
Description: Answers to Problem 7 answer sheet (MIE 304)

Example command line usage:
    $ python problem_7.py --q1 True
'''

import argparse
import os
from utils.load_data import load_csv

# Globals
Q6_DATA_FP = "../data/07_data.csv"
DATA = load_csv(Q6_DATA_FP)

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--q1', help='Run question 1')
args = parser.parse_args()

if args.q1:
    print(DATA)

    
