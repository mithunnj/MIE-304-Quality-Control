'''
Owner: Mit Jothiravi 1002321258
Description: Utils to calculate statistics for user
'''
import scipy.stats as stat
import numpy as np
import sys
import math

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



def mode_mean_median_calc(data):
    '''
    data: Represented as a pd dataframe
    '''

    mode = data.Densities.mode()
    mean = data.Densities.mean()
    median = data.Densities.median()

    return mode, mean, median


def IQR(data):
    '''
    data: Represented as a pd dataframe
    '''

    upper_quantile = data.Densities.quantile(0.75)
    lower_quantile = data.Densities.quantile(0.25)

    inter_quantile_range = upper_quantile - lower_quantile

    return inter_quantile_range

def binom(num_success, total_sample_size, prob_success):
    '''
    Inputs:
        - num_success: Number of successful outcomes in the binomial outcomes of the sample
        - total_sample_size: Total size of sample/trial
        - prob_success: Probability of success
    '''

    binom_result = stat.binom.pmf(num_success, total_sample_size, prob_success)

    return binom_result

def geometric(num_success, prob_failure):
    '''
    Inputs:
        - num_success: Number of successful outcomes before a failure
        - prob_failure: The pribability of failure (1-probability of success)
    '''

    geom_result = stat.geom.pmf(num_success, prob_failure)

    return geom_result

def hyper_geometric(num_failure_sample, pop_size, num_failure_pop, sample_size):
    '''
    Inpute:
        - num_failure_sample: Number of defective items we are calculating probability for in sample.
        - pop_size: Size of the population that our trial is drawing from, and that our statistics depend on.
        - num_failure_pop: Number of defective items that were stated in the population statistics.
        - sample_size: Size of the sample of our trial.
    '''

    hyper_geom_results = stat.hypergeom.pmf(num_failure_sample, pop_size, num_failure_pop, sample_size)

    return hyper_geom_results

def poisson(num_defects_per_unit, average_num_defects_per_unit):
    '''
    Input:
        - num_defects_per_unit: When the question asks you the probability of # defects per 
            unit of product, this is where you input it.
        - average_num_defects_per_unit: This should be given in the question as the mu (average) numbeer
            of defects in the system.
    '''

    poisson_results = stat.poisson.pmf(num_defects_per_unit, average_num_defects_per_unit)

    return poisson_results

def normal(request_data, population_mean, population_std):
    '''
    Inputs:
        - request_data: Probabilty information to compute for from the population statistics.
        - population_mean: Straight forward
        - population_std: Straight forward

    NOTE: This will return the area under the normal distribution graph for the area < request_data.
        If you want to determine probabilty of the event occuring for > request_data, you have to do 1 - return
    '''

    normal_results = stat.norm.cdf(request_data, population_mean, population_std)

    return normal_results

def null_hypothesis_testing_single_sample(df, column_name, null_hypothesis):
    '''
    Inputs:
        - df: The panda data frame that is loaded in (from a .csv file for example)
        - column_name: The columns name that we are parsing information for (ex. ‘Net Contents (Oz)’)
        - null hypothesis: The null hypothesis to test for given in the question.
    - Similar question: https://www.khanacademy.org/math/statistics-probability/significance-tests-one-sample/more-significance-testing-videos/v/hypothesis-testing-and-p-values
        - Approach: Hypothesis Test with P-value

    Use this for p-value testing for a single sample problem
    '''

    ## Step 1: Determine the Hypothesis
    ### Null Hypothesis: Population mean (mu) equals exactly 12
    ### Alternative Hypothesis: Population mean (mu) does not equal 12
    df_copy = df.copy() # Make a copy of the data frame to avoid changing the loaded df

    q1_filtered = df_copy[df_copy[column_name].notna()]
    q1_data = q1_filtered[column_name]
    mu_samp = q1_data.mean() # 12.004399999999999
    sig_samp = q1_data.std() # 0.02310844001658249
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
    print("\nQ1 Results: \n")
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

    return beta[0]

def r_chart_values(data, sample_size):
    '''
    Inputs:
        - data: pd.dataframe
        - sample_size: Per sample, how many values are there

    This will compute all the R-chart values that are required for X-bar and for plotting R-bar chart
    '''

    r_vals = list() # Store for the R values from all the samples

    # Step 1: Calculate and store the range of all the sample ranges
    for i in data['Sample Number'].unique().tolist():

        # Parse sample specific data
        ## NOTE: Make sure that you change the field that you want to parse the data for - in this case Voltage
        sample_data = data[data['Sample Number'] == i]['Voltage']

        # Fetch max/min value
        max_val = sample_data.max()
        min_val = sample_data.min()

        r_sample = max_val-min_val
        r_vals.append(r_sample) # This is to remove the weird numpy datastructure formatting

    # Step 2: Calculate R-bar
    r_bar = sum(r_vals)/len(r_vals) # (Sum of R values) / (Number of samples)

    # Step 3: Calculate Upper/Lower Control Limits
    UCL = D4[sample_size]*r_bar
    LCL = D3[sample_size]*r_bar

    return r_vals, r_bar, UCL, LCL
