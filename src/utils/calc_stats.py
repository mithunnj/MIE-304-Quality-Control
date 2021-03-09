'''
Owner: Mit Jothiravi 1002321258
Description: Utils to calculate statistics for user
'''
import scipy.stats as stat
import sys

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
        num_failure_sample: Number of defective items we are calculating probability for in sample.
        pop_size: Size of the population that our trial is drawing from, and that our statistics depend on.
        num_failure_pop: Number of defective items that were stated in the population statistics.
        sample_size: Size of the sample of our trial.
    '''

    hyper_geom_results = stat.hypergeom.pmf(num_failure_sample, pop_size, num_failure_pop, sample_size)

    return hyper_geom_results
