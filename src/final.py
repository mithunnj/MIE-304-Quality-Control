'''
Owner: Mit Jothiravi (1002321258)
Description: Answers to Problem 9 answer sheet (MIE 304)

Example command line usage:
    $ python problem_9.py --q1 True
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
DATA_FP = "../data/09_data.csv"
DATA = load_csv(DATA_FP)

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--q1', help='Run question 1')
args = parser.parse_args()

if args.q1:
    '''
    '''

