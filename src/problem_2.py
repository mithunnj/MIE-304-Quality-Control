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
from utils.calc_stats import mode_mean_median_calc, IQR, binom, geometric, hyper_geometric

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
parser.add_argument('--q6', help='Run question 6')
parser.add_argument('--q7', help='Run question 7')
parser.add_argument('--q8', help='Run question 8')
parser.add_argument('--q9', help='Run question 9')
parser.add_argument('--q10', help='Run question 10')
parser.add_argument('--q11', help='Run question 11')
args = parser.parse_args()

if args.q1:
    stem_plot(DATA)

if args.q2:
    bins = 6
    histogram_plot(DATA, bins)

if args.q3:
    bins = 6
    cumulative_frequency_plot(DATA, bins)

if args.q4:
    box_plot(DATA)

if args.q5:
    time_series_plot(DATA)

if args.q6:
    mode, mean, median = mode_mean_median_calc(DATA)
    print(" Mode: {}\n\n Median: {}\n Mean: {}".format(mode, median, mean))

if args.q7:
    iqr = IQR(DATA)
    print('Inter quantile range (IQR) is: {}'.format(iqr))

if args.q8:
    num_success_head = 5
    total_sample_size = 10
    percent_success_head = 0.5
    binom_result = binom(num_success_head, total_sample_size, percent_success_head)

    print('Biomial distribution result: {}'.format(binom_result))

if args.q9:
    num_success = 7
    total_sample_size = 10
    percent_success = 0.9
    binom_result = binom(num_success, total_sample_size, percent_success)

    print('Biomial distribution result: {}'.format(binom_result))

if args.q10:
    num_success = 5
    percent_fail = 0.05
    geom_result = geometric(num_success, percent_fail)

    print('Geometric distribution result: {}'.format(geom_result))

if args.q11:
    num_defects_sample = 1
    sample_size = 3
    num_defects_pop = 5
    pop_size = 20
    hyper_geom_result = hyper_geometric(num_defects_sample, pop_size, num_defects_pop, sample_size)

    print('Hyper-geometric distribution result: {}'.format(hyper_geom_result))



