'''
Owner: Mit Jothiravi 1002321258
Description: Utils to calculate statistics for user
'''

def mode_mean_median_calc(data):

    mode = data.Densities.mode()
    mean = data.Densities.mean()
    median = data.Densities.median()

    return mode, mean, median