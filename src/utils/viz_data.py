'''
Owner: Mit Jothiravi 1002321258
Description: Utils to visualize data with various plots
'''
import stemgraphic as stl
import matplotlib.pyplot as plt
import sys

def stem_plot(data):
    ret_plot, ret_axes = stl.stem_graphic(data.Densities, asc=False)
    plt.show()
    print("Successfully finished plotting Stem Plot")

    return

def histogram_plot(data, bins):
    if not bins:
        sys.exit('ERROR: <viz_data> assertion failed to find valid number of bins for Histogram')

    plt.hist(data.Densities,color='blue',bins=bins)
    plt.show()
    print("Successfully finished plotting Histogram with {} bins".format(bins))

    return

def cumulative_frequency_plot(data, bins):
    if not bins:
        sys.exit('ERROR: <viz_data> assertion failed to find valid number of bins for Histogram')

    plt.hist(data.Densities,color='pink',bins=bins,cumulative=True)
    plt.show()
    print("Successfully finished plotting Cumulative Frequency with {} bins".format(bins))

    return

