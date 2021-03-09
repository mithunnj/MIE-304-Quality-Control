'''
Owner: Mit Jothiravi 1002321258
Description: Utils to visualize data with various plots
'''
import stemgraphic as stl
import matplotlib.pyplot as plt
import sys

def viz_data(data, stem_plot=False, histogram=False, bins=None, cumulative_frequency=False):

    # Generate a stem_plot
    if stem_plot:
        ret_plot, ret_axes = stl.stem_graphic(data.Densities, asc=False)
        plt.show()
        print("Successfully finished plotting Stem Plot")

    # Generate a Historgram with user defined number of bins
    if histogram:
        if not bins:
            sys.exit('ERROR: <viz_data> assertion failed to find valid number of bins for Histogram')

        plt.hist(data.Densities,color='blue',bins=bins)
        plt.show()
        print("Successfully finished plotting Histogram with {} bins".format(bins))

    # Generate a Cumulative Frequency plot
    if cumulative_frequency:
        if not bins:
            sys.exit('ERROR: <viz_data> assertion failed to find valid number of bins for Histogram')

        plt.hist(data.Densities,color='pink',bins=bins,cumulative=True)
        plt.show()
        print("Successfully finished plotting Cumulative Frequency with {} bins".format(bins))




    return