'''
Owner: Mithun Jothiravi (1002321258)
Description: Code submission for Midterm Exam: March 11, 2021

Example command line usage:
    $ python midterm.py --q1 True
'''

import argparse
import os
from utils.load_data import load_csv

# Globals
DATA_FP = "../data/<CHANGE THIS>"
DATA = load_csv(DATA_FP)

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--q5', help='Run question 5')
args = parser.parse_args()