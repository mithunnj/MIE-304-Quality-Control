'''
Owner: Mit Jothiravi (1002321258)
Description: Answers to Problem 12 answer sheet (MIE 304)

Example command line usage:
    $ python problem_12.py --q1 True
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
DATA_FP = "../data/12_data.csv"
DATA = load_csv(DATA_FP)

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--q1', help='Run question 1')
args = parser.parse_args()

if args.q1:
    '''
    This question addresses Q1
    '''
    print(DATA)
    