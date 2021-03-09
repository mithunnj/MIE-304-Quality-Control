'''
Owner: Mit Jothiravi 1002321258
Description: Utils to calculate statistics for user
'''

def mode_mean_median_calc(data):

    mode = data.Densities.mode()
    mean = data.Densities.mean()
    median = data.Densities.median()

    return mode, mean, median


def IQR(data):

    upper_quantile = data.Densities.quantile(0.75)
    lower_quantile = data.Densities.quantile(0.25)

    inter_quantile_range = upper_quantile - lower_quantile

    return inter_quantile_range
