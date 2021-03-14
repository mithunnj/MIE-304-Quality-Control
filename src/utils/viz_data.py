'''
Owner: Mit Jothiravi 1002321258
Description: Utils to visualize data with various plots
'''
import stemgraphic as stl
import matplotlib.pyplot as plt
import numpy as np
import sys

def stem_plot(data):
    '''
    data: Represented as a pd dataframe
    '''
    ret_plot, ret_axes = stl.stem_graphic(data.Densities, asc=False)
    plt.show()
    print("Successfully finished plotting Stem Plot")

    return

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

def cumulative_frequency_plot(data, bins):
    '''
    data: Represented as a pd dataframe
    bins: # of sub bins to store the histogram buckets
    '''
    if not bins:
        sys.exit('ERROR: <viz_data> assertion failed to find valid number of bins for Histogram')

    plt.hist(data.Densities,color='pink',bins=bins,cumulative=True)
    plt.show()
    print("Successfully finished plotting Cumulative Frequency with {} bins".format(bins))

    return

def box_plot(data):
    '''
    data: Represented as a pd dataframe
    '''
    plt.boxplot(data.Densities)
    plt.grid(True)
    plt.show()
    print("Successfully finished plotting Box Plot.")

    return

def time_series_plot(data):
    '''
    data: Represented as a pd dataframe
    '''
    y = [i[0] for i in data.values.tolist()]
    x = [i for i in range(1, len(y)+1)]

    plt.scatter(x, y)
    plt.show()

    print('Succesfully created Scatter Plot')

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


