'''
Owner: Mit Jothiravi (1002321258)
Description: Answers to Problem 3 answer sheet (MIE 304)

Example command line usage:
    $ python problem_3.py --q1 True
'''

import argparse
import os

from utils.load_data import load_csv
from utils.calc_stats import null_hypothesis_testing, type_2_beta

# Globals
Q3_DATA_FP = "../data/03_data.csv"
DATA = load_csv(Q3_DATA_FP)

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--q1', help='Run question 1')
parser.add_argument('--q2', help='Run question 2')
parser.add_argument('--q3', help='Run question 3')
parser.add_argument('--q4', help='Run question 4')
args = parser.parse_args()

if args.q1:
    null_hypothesis = 12.00 # Given in question
    null_hypothesis_testing(DATA, 'Net Contents (Oz)', null_hypothesis)

if args.q2:
    given_population_mean = 12.00
    proposed_population_mean = 12.012 # We are asked to calculate the probability of failure for this value
    sample_size = 100

    beta_probability = type_2_beta(DATA, given_population_mean, proposed_population_mean, sample_size)

    print('The probability that we fail to reject mu = {}, is Beta = {}'.format(proposed_population_mean, beta_probability))
    
