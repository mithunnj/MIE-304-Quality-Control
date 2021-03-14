'''
Owner: Mithun Jothiravi (1002321258)
Description: Code submission for Midterm Exam: March 11, 2021

Example command line usage:
    $ python midterm.py --q1 True
    $ python midterm.py --q2 True --q3 True --q4 True --q5 True
'''
import argparse
import os
import sys
import math
import scipy.stats as stat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stemgraphic as stl

##### Utils helper functions
def histogram_plot(data, bins):
    '''
    data: Represented as a pd dataframe
    bins: # of sub bins to store the histogram buckets
    '''
    if not bins:
        sys.exit('ERROR: <viz_data> assertion failed to find valid number of bins for Histogram')

    plt.hist(data, color='blue',bins=bins)
    plt.show()
    print("Successfully finished plotting Histogram with {} bins".format(bins))

    return

def x_bar_any_chart_plot(num_samples, LIST_VAL, CL, LCL, UCL, chart_name):
    '''
    Input: 
        - num_samples: The number of total samples taken in this trial (m). NOTE: This isn't the size of an individual sample.
        - LIST_VAL: List of values (ex. X-bar values for X-bar chart, or R values for all samples for R-chart)
        - CL, UCL, LCL: Control limits for the respective charts pre-computed
        - chart_name: String representing the name of the chart for the plot (ex. X-bar, R)
    '''

    # Generate straight lines for the Control Limits
    ucl_line=np.full(num_samples, UCL)
    lcl_line=np.full(num_samples, LCL)
    cl_line=np.full(num_samples, CL)

    # Generate enough space for number of samples taken
    x=list(range(0,num_samples))

    plt.plot(LIST_VAL, marker="o")
    plt.plot(ucl_line, color='k')
    plt.plot(lcl_line, color='k')
    plt.plot(cl_line, color='k')
    plt.xticks(x) #Creates x axis ticks
    plt.grid(True)
    plt.title('{} chart'.format(chart_name))
    plt.show()

    return

def load_csv(fp):
    '''
    Input: fp <str> - The filepath to the .csv data file
    
    Take in a .csv filepath, and convert it to a parseable pd dataframe.
    '''

    # Check that filepath exists
    if not os.path.isfile(fp):
        sys.exit('ERROR: <load_csv> assertion failed to find filepath: {}'.format(fp))

    # Check if file is a .csv
    if not fp.endswith('.csv'):
        sys.exit('ERROR: <load_csv> assertion failed to find .csv file : {}'.format(fp))

    # Load, convert .csv into Panda datastructure
    data = pd.read_csv(fp)

    return data

def mode_mean_median_calc(data):
    '''
    data: Represented as a pd dataframe
    '''

    mode = data.mode()
    mean = data.mean()
    median = data.median()

    return mode, mean, median

def IQR(data):
    '''
    data: Represented as a pd dataframe
    '''

    upper_quantile = data.quantile(0.75)
    lower_quantile = data.quantile(0.25)

    inter_quantile_range = upper_quantile - lower_quantile

    return inter_quantile_range

def r_chart_values(data, sample_size, category_name):
    '''
    Inputs:
        - data: pd.dataframe
        - sample_size: Per sample, how many values are there
        - category_name: The column of the dataframe that you want to parse

    This will compute all the R-chart values that are required for X-bar and for plotting R-bar chart
    '''

    r_vals = list() # Store for the R values from all the samples

    # Step 1: Calculate and store the range of all the sample ranges
    for i in data['Sample'].unique().tolist():

        # Parse sample specific data
        sample_data = data[data['Sample'] == i][category_name]

        # Fetch max/min value
        max_val = sample_data.max()
        min_val = sample_data.min()

        r_sample = max_val-min_val
        r_vals.append(r_sample) # This is to remove the weird numpy datastructure formatting

    # Step 2: Calculate R-bar (This will act like the CL for R-chart)
    r_bar = sum(r_vals)/len(r_vals) # (Sum of R values) / (Number of samples)

    # Step 3: Calculate Upper/Lower Control Limits
    UCL = D4[sample_size]*r_bar
    LCL = D3[sample_size]*r_bar

    return r_vals, r_bar, UCL, LCL

def x_bar_r_chart_values(data, sample_size, R_BAR, category_name):
    '''
    Inputs:
        - data: pd.dataframe
        - sample_size: Per sample, how many values are there
        - R_BAR: This is the centre line from the R chart that should be computed before this
        - category_name: The column of the dataframe that you want to parse

    This will compute all the R-chart values that are required for X-bar and for plotting R-bar chart
    '''
    x_bar_vals = list() # Store for the X_bar values from all the samples

    # Step 1: Calculate and store the range of all the sample ranges
    for i in data['Sample'].unique().tolist():

        # Parse sample specific data
        ## NOTE: Make sure that you change the field that you want to parse the data for - in this case Voltage
        sample_data = data[data['Sample'] == i][category_name]

        # Calculate average of sample (x_bar)
        x_bar = sample_data.mean()
        x_bar_vals.append(x_bar)


    # Step 2: Calculate X_bar_bar (Average of all sample averages)
    x_bar_bar = sum(x_bar_vals) / len(x_bar_vals) 

    # Step 3: Calculate Upper/Lower Control Limits
    UCL = x_bar_bar + A2[sample_size]*R_BAR
    LCL = x_bar_bar - A2[sample_size]*R_BAR

    return x_bar_vals, x_bar_bar, UCL, LCL

def chart_control_type(chart_name, param_vals, param_UCL, param_LCL):
    '''
    Input:
        - chart_name: Specify the type of chart that we are checking for. (ex. X-bar, R, S, etc.)
        - param_vals: All the computed values for the specific chart type, for the sample process
        - param_UCL: Upper control limit
        - param_LCL: Lower control limit

    Given the chart values, check if the process is in control. It will just print the results to the console.
    '''

    # Verify that parameters do not fall out of the control limits
    for param in param_vals:
        if param < param_LCL or param > param_UCL:
            print("FAILURE: {} chart is out of control because Val: {}, falls out of control limits: LCL {}, UCL {}\n".format(chart_name, param, param_LCL, param_UCL))
            return

    
    print("PASS: {} chart is in control\n".format(chart_name))

    return

def null_hypothesis_testing(df, column_name, null_hypothesis, sample_mean=None, sample_std=None):
    '''
    Inputs:
        - df: The panda data frame that is loaded in (from a .csv file for example)
        - column_name: The columns name that we are parsing information for (ex. ‘Net Contents (Oz)’)
        - null hypothesis: The null hypothesis to test for given in the question.
    - Approach: Hypothesis Test with P-value

    Use this for p-value testing for a single sample problem
    '''

    ## Step 1: Determine the Hypothesis
    ### Null Hypothesis: Population mean (mu) equals exactly 12
    ### Alternative Hypothesis: Population mean (mu) does not equal 12
    df_copy = df.copy() # Make a copy of the data frame to avoid changing the loaded df

    q1_filtered = df_copy[df_copy[column_name].notna()]
    q1_data = q1_filtered[column_name]
    mu_samp = sample_mean if sample_mean else q1_data.mean() 
    sig_samp =  sample_std if sample_std else q1_data.std() 
    total_sample = len(q1_data) # 25, with row 26 removed due to NaN

    ## Step 2: Prove Null Hypothesis
    ### Determine population mean (mu_pop) and population standard deviation (sig_pop)
    ###     from sample mean (mu_samp) and sample standard deviation (sig_samp)
    mu_pop = null_hypothesis # Null Hypothesis from question
    sig_pop = sig_samp/(math.sqrt(total_sample)) # An approximation of the pop. std deviation

    ## Step 3: Calculate the t-score
    ### (mu_pop - mu_samp) / sig_pop
    t_score = abs((mu_pop - mu_samp)/sig_pop)

    ## Step 4: Calculate p-value from T score as outlined here: https://stackoverflow.com/questions/23879049/finding-two-tailed-p-value-from-t-distribution-and-degrees-of-freedom-in-python
    p_value = stat.t.sf(t_score, total_sample - 1)*2 #twosided

    ## Step 5: Validate null hypothesis
    ### Based on this p-value wiki: https://www.google.com/search?q=p-value+threshold+for+hypothesis+test&oq=p-value+threshold+for+hypothesis+test&aqs=chrome..69i57j33i22i29i30i395l7.7110j1j7&sourceid=chrome&ie=UTF-8
    ###     "Usage. The p-value is widely used in statistical hypothesis testing, specifically in null hypothesis significance testing. ... For typical analysis, using the 
    ###     standard α = 0.05 cutoff, the null hypothesis is rejected when p < .05 and not rejected when p > .05."
    print("\nQ4 Results: \n")
    print("Null Hypothesis REJECTED because of p-value") if (p_value < 0.05) else print("Null Hypothesis NOT REJECTED because of p-value")
    print("\n Extra stats: \
        \n\t Mu Pop. (Mean): {}\
        \n\t Sig Pop. (Std): {}\
        \n\t Mu Sample (Mean): {}\
        \n\t Sig Sample (Std): {}\
        \n\t Sample size: {}\
        \n\t T-score: {}\
        \n\t p_value: {} \
        ".format(mu_pop, sig_pop, mu_samp, sig_samp, total_sample, t_score, p_value))

    return

def type_2_beta(data, mu0, mu1, n):
    '''
    Inputs:
        - data: pd dataframe
        - mu0: The given population mean in the problem.
        - mu1: This is the mu that is proposed that we are trying to find the probability of rejection.
        - n: Sample size
    
    Used to calculate Beta: b = P{type II error} = P{fail to reject H0 |H0 is false}
    '''

    delta = mu1 - mu0
    alpha = 0.05 # This is the standard cutoff value that we use for Null Hypothesis

    s = data.std()
    z_alpha2 = stat.norm.ppf(1-alpha/2)

    beta = stat.norm.cdf((z_alpha2 - (delta*np.sqrt(n))/s),0,1) - stat.norm.cdf((-z_alpha2 - (delta*np.sqrt(n))/s),0,1) # This returns an array of Beta values

    # Based on the solutions from the assignments, the first index seems to be it.

    return beta


##### Globals
DATA_FP = "../data/midterm_data.csv"
DATA = load_csv(DATA_FP)
# Create Table of Constants for Control Charts
A2 = {
    2: 1.880, 
    3: 1.023,
    4: 0.729, 
    5: 0.577, 
    6: 0.483, 
    7: 0.419,
    8: 0.373,
    9: 0.337,
    10: 0.308
    }

A3 = {
    2: 2.659, 
    3: 1.954,
    4: 1.628, 
    5: 1.427, 
    6: 1.287, 
    7: 1.182,
    8: 1.099,
    9: 1.032,
    10: 0.975
}

B3 = {
    2: 0, 
    3: 0,
    4: 0, 
    5: 0, 
    6: 0.030, 
    7: 0.118,
    8: 0.185,
    9: 0.239,
    10: 0.284
    }

B4 = {
    2: 3.267, 
    3: 2.568,
    4: 2.266, 
    5: 2.089, 
    6: 1.970, 
    7: 1.882,
    8: 1.815,
    9: 1.761,
    10: 1.716
    }
    

D3 = {
    2: 0, 
    3: 0,
    4: 0, 
    5: 0, 
    6: 0, 
    7: 0.076,
    8: 0.136,
    9: 0.184,
    10: 0.223
    }

D4 = {
    2: 3.267, 
    3: 2.574,
    4: 2.282, 
    5: 2.114, 
    6: 2.004, 
    7: 1.924,
    8: 1.864,
    9: 1.816,
    10: 1.777
    }


##### Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--q2', help='Run question 2')
parser.add_argument('--q3', help='Run question 3')
parser.add_argument('--q4', help='Run question 4')
parser.add_argument('--q5', help='Run question 5')
args = parser.parse_args()


##### Specific questions start below
if args.q2:
    # Step 1: Fetch the Volume Data
    vol_data = DATA["Volume"]

    # Q2 a) Calculate mean information
    _, mean, _ = mode_mean_median_calc(vol_data)
    print("Q2 a): Mean of volume is {}\n".format(mean))

    # Q2 b) Calculate range
    max_val = vol_data.max()
    min_val = vol_data.min()
    range_val = max_val - min_val
    print("Q2 b): Range of volume is {}\n".format(range_val))

    # Q2 c) Calculate IQR
    IQR_val = IQR(vol_data)
    print("Q2 c) Inter-quantile range is {}\n".format(IQR_val))

    # Q2 d) Plot histogram
    bins = round(vol_data.max()) - round(vol_data.min()) # Determine bins based on the range of Volume values
    histogram_plot(vol_data, bins)

if args.q5:
    df = DATA[['Sample', 'Voltage']].dropna() # Extract only the Sample and Voltage data
    sample_size = 4 # In each sample, there are 4 values 

    # Calculate R-chart values
    R_VALS, R_BAR, R_UCL, R_LCL = r_chart_values(df, sample_size, 'Voltage')

    # Calculate X-bar chart values
    X_BAR_VALS, X_BAR_BAR, X_BAR_UCL, X_BAR_LCL = x_bar_r_chart_values(df, sample_size, R_BAR, 'Voltage')

    # Check if R-chart & X-bar chart are in control
    chart_control_type('R', R_VALS, R_UCL, R_LCL)
    chart_control_type('X-bar', X_BAR_VALS, X_BAR_UCL, X_BAR_LCL)

    # Calculate total number of samples takes (m)
    m = len(df["Sample"].unique().tolist())
 
    # Plot X-bar chart
    x_bar_any_chart_plot(m, X_BAR_VALS, X_BAR_BAR, X_BAR_LCL, X_BAR_UCL, 'X-bar')

    # Plot R chart
    x_bar_any_chart_plot(m, R_VALS, R_BAR, R_LCL, R_UCL, 'R')

    # Q5 b) Print out the control limits
    print("\nControl Limits: \n")
    print("\nX-Chart:\n CL = {}\n LCL = {}\n UCL = {}\n".format(X_BAR_BAR, X_BAR_LCL, X_BAR_UCL))
    print("\nR Chart:\n CL = {}\n LCL = {}\n UCL = {}".format(R_BAR, R_LCL, R_UCL))

    # Q5 c) Process mean and std
    d2 = 2.059 # For sample size = 4
    sigma = R_BAR/d2

    print("\nQ5 c):\n Process std. approx: {}\n Process mean approx: {}".format(sigma, X_BAR_BAR))

    # Q5 d) Calculate Beta value to answer this
    n = 3 # Detecting the shift in the third sample
    mu_1 = 13.5 # Shifting the process mean to 13.5 V
    mu_0 = X_BAR_BAR # Previous process mean 

    k = ( (mu_1 - mu_0) )/ sigma

    beta = stat.norm.cdf(3-k*np.sqrt(n),0,1)-stat.norm.cdf(-3-k*np.sqrt(n),0,1)

    print('\nQ5 d) Calculating Beta value results')
    print(' Beta = ', beta)
    print(' Prob of detecting shift on third sample = ', 1-beta)

    # Q5 e) Calculate Average Run Length (ARL) using previous Beta value
    ARL = 1 / (1 - beta)

    print('\nQ5 e) Average Run Length using previous Beta value is:')
    print(' Beta = ', beta)
    print(' ARL = ', ARL)

if args.q4:
    # Q4 a) Perform a Null Hypothesis test
    column_name = 'Volume'
    null_hypothesis = 3.0 # I changed the 30mL to 3.0, because how does your values correlate with the data ?
    sample_mean = 2.989 # I changed 29.89mL to 2.989, for the same reason above.
    sample_std = 0.036 # I changed the 0.36 mL to 0.036, for the same reason as above.

    # NOTE: Uncomment line below to get the results for Q4 a)
    #null_hypothesis_testing(DATA, column_name, null_hypothesis, sample_mean=sample_mean, sample_std=sample_std)

    # Q4 b) Calculate Beta to determine failure to reject

    beta = type_2_beta(DATA['Volume'], DATA['Volume'].mean(), 2.98, 100)

    print('\nQ4 b) Calculate Beta to determine Prob. Failure to Reject:')
    print(" Beta = {}".format(beta))

if args.q3:
    x = 13.1
    mean = 11.6818
    std = 0.5

    # Question 
    prob_calc = 1 - stat.norm.cdf(x, mean, std)
    print('\nQ3 a) :')
    print('\nProbability: {}'.format(prob_calc))

    prob_3_one_hour = stat.poisson.pmf(3,4)

    print('\nQ3 b) :')
    print(prob_3_one_hour) # Probability will have one one-hour period with 3 orders.
    print(prob_3_one_hour**4) # Probability they will have 4 one-hour periods with 3 orders.
    print(stat.geom.pmf(5,0.16803135574154085)) # Probabiltiy they will have 5 successes till teh 5th hour of operation (a success is defined as a 3 order 1 hour period).



    


    


