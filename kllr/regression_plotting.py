import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from .regression_model import kllr_model, calculate_weigth

'''
Plotting Params
'''
import matplotlib as mpl

mpl.rcParams['xtick.direction'], mpl.rcParams['ytick.direction'] = 'in', 'in'
mpl.rcParams['xtick.major.size'], mpl.rcParams['xtick.minor.size'] = 14, 8
mpl.rcParams['xtick.major.width'], mpl.rcParams['xtick.minor.width'] = 1.2, 0.8
mpl.rcParams['xtick.major.pad'], mpl.rcParams['xtick.minor.pad'] = 10, 10
mpl.rcParams['ytick.major.size'], mpl.rcParams['ytick.minor.size'] = 14, 8
mpl.rcParams['ytick.major.width'], mpl.rcParams['ytick.minor.width'] = 1.2, 0.8
mpl.rcParams['xtick.labelsize'], mpl.rcParams['ytick.labelsize'] = 20, 20
default_cmap = plt.cm.viridis

'''
Initiate Dataset:
    All variables can be accesses and modified as seen fit by user.
    eg. Class.fontsize.xlabel = 0
    eg. Class.Colors = ['red', 'blue', 'aqua']
'''

# Dictionary that allows user to change fontsize of plots whenever necessary
fontsize = pd.Series({
    'title': 25,
    'legend': 18,
    'xlabel': 25,
    'ylabel': 25
})

'''
Plotting functions

-------------------
Disclaimer:
-------------------

(i)   In general, any function that does a split in a 3rd variable only includes data with x_data > cutoff.

(ii)  Functions that do NOT do a split include data from x_data > cutoff - 0.5.

(iii) The choice in (ii) is needed because LLR uses all the data, and so not including data beneath cutoff value
      introduces artifacts at the x-value boundary

(iv)  However, including the x_data > cutoff - 0.5 clause in the split version will mess up how we split the data according to the 3rd variable.
      We split the data into bins in split_variable, where each bin contains an equal number of halos.
      So including the halos below our cutoff will change how our bin-edges are determined and thus affect our results.

Parameters
-------------

df : pandas dataframe
    DataFrame containing all properties

xlabel, ylabel(s) : str
    labels of the data vectors of interest in the dataframe.
    In case of covariance/correlation matrix functions, we pass a list of labels into the "ylabels" parameter.

show_data : boolean
    Used in Plot_Fit function to show the datapoints used to make the LLR fit.
    Is set to show_data = False, by default

nbins : int
    Sets the number of points within x_range that the parameters are sampled at.
    When plotting a PDF it sets the number of bins the PDF is split into.

xrange : list, tuple, np.array
    A 2-element list, tuple, or numpy array that sets the range of x-values for which we compute and plot parameters.
    By default, xrange = None, and the codes will choose np.min(x_data) and np.max(x_data) as lower and upper bounds.

nBootstrap : int
    Sets how many bootstrap realizations are made when determining statistical error in parameters.

percentile : list, tuple, np.array
    List, tuple, or numpy array whose values set the bounds of parameter distribution to be plotted when plotting uncertainties.
    Assuming gaussianity for the distributions, a 1sigma bound can be gained using [16., 84.], which is also the default value.

split_label : str
    Label of the data vector used to split the data, or condition the data, on a secondary variable.

split_bins : int, list, tuple, numpy array
    Can be either number of bins (int), or array of bin_edges.

    If an int is provided the modules will determine the bin edges themselves using the data vector.
    By default, edges are set so there are equal number of data points in each bin.
    Note that the bin edges in this case will be determed using all data passed into the function. However,
    the plotting and computations will be done only using data with x-values within the bounds set by the xrange parameter.

    If a list is provided then the list elements serve as the bin edges

split_mode : str
    Sets how the data is split/conditioned based on the split variable
    If 'Data', then all halos are binned based on the variable df[split_label]
    If 'Residuals', then we fit split_label vs. xlabel, then split the data into bins based on the residual values

labels : list of str
    Allows for user-defined labels for x-axis, y-axis, legend labels.

funcs : dictionary
    Available when plotting PDF. A dictionary of format {key: function}, where the function will be run on all the residuals in
    every bootstrap realization. Results for median values and 1sigma bounds will be printed and stored in the Output_Data array

verbose : boolean
    Controls the verbosity of the model's output.

Returns
----------

Dictionary
    A dictionary containing all outputs. Is of the form {parameter_name : numpy-array}
    The keys are parameter names (eg. x, y, slope, scatter), and the values
    are the computed properties themselves. Any data shown in plots will, and should, be stored in the Output_data dict.

    In the split case, the dictionary will be a 2D dictionary of form {Bin_id : {parameter_name : array}}, where Bin_id
    represents the bin, determined by split_variable, within which the parameters were computed

Matplotlib.axes
    Axes on which results were plotted

## TODO:
    * give an error : data, ax = Plot_Fit_Split(df, 'M200', 'MStar_BCG100', 'z_form', split_bins=3, split_mode='Residuals')
    * What to do if df labels are int instead of str (eg. 1, 2, 3 etc.)
'''

# constant (set it to np.log(10.0) if you wish to go from dex to fractional error in scatter)
Ln10 = np.log(10.0) #1.0

def setup_color(color, split_bins, cmap=None):
    """
    Takes a list of colors, and the number of split_bins and a color map to generate a color map for each split bin.
    if color is None it use cmap to generate a list of colors.

    Parameters
    ----------
    color : list, or None
        a list of matplotlib colors or None

    split_bins : int or list
        number of split bins or the boundary of split bins

    cmap : cmap
        if color=None, It takes cmap and generate a list of matplotlib colors.

    Returns
    -------
    a list of matplotlib colors
    """

    if color is not None:
        if isinstance(split_bins, int):
            if len(color) < split_bins:
                raise ValueError('len(color) is less than split bins while len(color) and'
                                 ' split bins must have the same length.')
        elif isinstance(split_bins, (np.ndarray, list, tuple)):
            if len(color) < len(split_bins) - 1:
                raise ValueError('len(color) is less than len(split_bins)-1 while '
                                 'len(color) must be larger than split bins.')

    if cmap is None:
        cmap = default_cmap

    if color is None:
        if isinstance(split_bins, int):
            color = cmap(np.linspace(0, 1, split_bins))
        elif isinstance(split_bins, (np.ndarray, list, tuple)):
            color = cmap(np.linspace(0, 1, len(split_bins) - 1))

    return color

def Plot_Fit(df, xlabel, ylabel, nbins=25, xrange=None, show_data=False,
             kernel_type='gaussian', kernel_width=0.2, xlog=False, ylog=False, ls = '-',
             color=None, labels=None, ax=None):

    lm = kllr_model(kernel_type, kernel_width)

    if ax == None:
        ax = plt.figure(figsize=(12, 8))

    plt.grid()

    if xlog: plt.xscale('log')
    if ylog: plt.yscale('log')

    if labels is None:
        labels = [xlabel, ylabel]

    output_Data = {}

    x_data, y_data = np.array(df[xlabel]), np.array(df[ylabel])

    mask = np.invert(np.isinf(x_data)) & np.invert(np.isinf(y_data))

    x_data, y_data = x_data[mask], y_data[mask]

    x, y = lm.fit(x_data, y_data, xrange=xrange, nbins=nbins)[0:2]

    if xlog: x = 10 ** x
    if ylog: y = 10 ** y

    # Add black line around regular line to improve visibility
    plt.plot(x, y, lw=6, color = 'k', label="", ls = ls)
    plt.plot(x, y, lw=3, color = color, ls = ls)

    output_Data['x'] = x
    output_Data['y'] = y

    # Code for bootstrapping the <y | x> values.
    # We don't use it since the uncertainty is very small in our case (not visible actually)

    if show_data:

        #Generate Mask so raw data is shown only for values of x_data within xrange
        if xrange == None:
            #If no xrange inputted, mask selected all available data
            Mask = np.ones(x_data.size, bool)
        else:
            Mask = (x_data > xrange[0]) & (x_data < xrange[1])

        x_data, y_data = x_data[Mask], y_data[Mask]

        if xlog: x_data = 10 ** x_data
        if ylog: y_data = 10 ** y_data
        plt.scatter(x_data, y_data, s=30, alpha=0.3, color = color, label="")

    plt.xlabel(labels[0], size=fontsize.xlabel)
    plt.ylabel(labels[1], size=fontsize.ylabel)

    return output_Data, ax

def Plot_Fit_Split(df, xlabel, ylabel, split_label, split_bins=[], nbins=25, xrange=None,
                   show_data=False, split_mode='Data', kernel_type='gaussian', kernel_width=0.2,
                   xlog=False, ylog=False, color=None, labels=None, ax=None):

    check_attributes(split_bins=split_bins, split_mode=split_mode)

    lm = kllr_model(kernel_type, kernel_width)

    if ax is None:
        ax = plt.figure(figsize=(12, 8))

    color = setup_color(color, split_bins, cmap=None)

    plt.grid()

    if xlog: plt.xscale('log')
    if ylog: plt.yscale('log')

    # If 3 labels not inserted, default to column names
    if labels is None:
        labels = [xlabel, ylabel, split_label]

    x_data, y_data, split_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[split_label])

    mask = np.invert(np.isinf(x_data)) & np.invert(np.isinf(y_data)) & np.invert(np.isinf(split_data))

    x_data, y_data, split_data = x_data[mask], y_data[mask], split_data[mask]

    # Choose bin edges for binning data
    # Choose bin edges for binning data
    if (isinstance(split_bins, int)):
        if split_mode == 'Data':
            split_bins = [np.percentile(split_data, float(i / split_bins) * 100) for i in range(0, split_bins + 1)]
        elif split_mode == 'Residuals':
            split_res = lm.calculate_residual(x_data, split_data, xrange=xrange)
            split_bins = [np.percentile(split_res, float(i / split_bins) * 100) for i in range(0, split_bins + 1)]
            split_res = lm.calculate_residual(x_data, split_data)

    # Need to compute residuals if split_mode == 'Residuals' is chosen
    elif isinstance(split_bins, (np.ndarray, list, tuple)) & (split_mode == 'Residuals'):
        split_res = lm.calculate_residual(x_data, split_data)

    # Define dictionary that will contain values that are being plotted
    # First define it to be a dict of dicts whose first level keys are split_bin number
    output_Data = {'Bin' + str(i): {} for i in range(len(split_bins) - 1)}

    # Loop over bins in split_variable
    for i in range(len(split_bins) - 1):

        # Mask dataset based on raw value or residuals to select only halos in this bin
        if split_mode == 'Data':
            split_mask = (split_data <= split_bins[i + 1]) & (split_data > split_bins[i])
        elif split_mode == 'Residuals':
            split_mask = (split_res < split_bins[i + 1]) & (split_res > split_bins[i])

        # Run LLR using JUST the subset
        x, y = lm.fit(x_data[split_mask], y_data[split_mask], xrange=xrange, nbins=nbins)[0:2]

        # Format label depending on Data or Residuals mode
        if split_mode == 'Data':
            label = r'$%0.2f <$ %s $< %0.2f$' % (split_bins[i], labels[2], split_bins[i + 1])
        elif split_mode == 'Residuals':
            label = r'$%0.2f < {\rm res}($%s$) < %0.2f$' % (split_bins[i], labels[2], split_bins[i + 1])

        if xlog: x = 10 ** x
        if ylog: y = 10 ** y

        # Add black line first beneath actual line to enhance visibility
        plt.plot(x, y, lw=6, color = 'k', label="")
        plt.plot(x, y, lw=3, color = color[i], label=label)

        # Store data to be outputted later
        output_Data['Bin' + str(i)]['x'] = x
        output_Data['Bin' + str(i)]['y'] = y

        if show_data:

            if xlog:
                x_data_tmp = 10 ** x_data
            else:
                x_data_tmp = x_data
            if ylog:
                y_data_tmp = 10 ** y_data
            else:
                y_data_tmp = y_data

            #Use the outputted 'x-vals' as min and max range
            mask = (x_data_tmp <= x[-1]) & (x_data_tmp >= x[0])

            # Only display data above our cutoff and of halos within the bins in split_data
            plt.scatter(x_data_tmp[mask & split_mask], y_data_tmp[mask & split_mask],
                        s=30, alpha=0.3, color = color[i], label="")

    plt.xlabel(labels[0], size=fontsize.xlabel)
    plt.ylabel(labels[1], size=fontsize.ylabel)
    plt.legend(fontsize=fontsize.legend)

    return output_Data, ax

def Plot_Fit_Params(df, xlabel, ylabel, nbins=25, xrange=None, nBootstrap=100,
                    kernel_type='gaussian', kernel_width=0.2, percentile=[16., 84.],
                    xlog=False, labels=None, color=None, verbose=True, ax=None, ls = '-'):

    lm = kllr_model(kernel_type, kernel_width)

    if ax is None:
        fig, ax = plt.subplots(2, figsize=(12, 10), sharex=True)
        fig.subplots_adjust(hspace=0.05)

    # Set x_scale to log. Leave y_scale as is.
    if xlog:
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')

    ax[0].grid()
    ax[1].grid()

    if labels is None:
        labels = [xlabel, ylabel]

    # Dictionary to store output values
    output_Data = {}

    # Load and mask data
    x_data, y_data = np.array(df[xlabel]), np.array(df[ylabel])

    mask = np.invert(np.isinf(x_data)) & np.invert(np.isinf(y_data))

    x_data, y_data = x_data[mask], y_data[mask]

    # Generate new arrays to store params in for each Bootstrap realization
    scatter = np.empty([nBootstrap, nbins])
    slope = np.empty([nBootstrap, nbins])
    intercept = np.empty([nBootstrap, nbins])

    if verbose:
        iterations_list = tqdm(range(nBootstrap))
    else:
        iterations_list = range(nBootstrap)

    for iBoot in iterations_list:

        # First bootstrap realization is always just raw data
        if iBoot == 0:
            xx, yy = x_data, y_data
        # All other bootstraps have shuffled data
        else:
            xx, index = lm.subsample(x_data)
            yy = y_data[index]

        # xline is always the same regardless of bootstrap so don't need 2D array for it.
        # yline is not needed for plotting in this module so it's a 'dummy' variable
        xline, yline, intercept[iBoot, :], slope[iBoot, :], scatter[iBoot, :] = lm.fit(xx, yy,
                                                                                       xrange=xrange,
                                                                                       nbins=nbins)[:5]
    if xlog: xline = 10 ** xline

    p = ax[0].plot(xline, np.mean(slope, axis=0), lw=3, color=color, ls = ls)
    color = p[0].get_color()
    ax[0].fill_between(xline, np.percentile(slope, percentile[0], axis=0), np.percentile(slope, percentile[1], axis=0),
                       alpha=0.4, label=None, color=color)
    ax[1].plot(xline, np.mean(scatter, axis=0) * Ln10, lw=3, color=color, ls = ls)
    ax[1].fill_between(xline, np.percentile(scatter, percentile[0], axis=0) * Ln10, np.percentile(scatter, percentile[1], axis=0) * Ln10,
                       alpha=0.4, label=None, color=color)

    # Output Data
    output_Data['x'] = xline

    output_Data['slope'] = np.median(slope, axis=0)
    output_Data['slope-'] = np.percentile(slope, percentile[0], axis=0)
    output_Data['slope+'] = np.percentile(slope, percentile[1], axis=0)

    # Output data for scatter (in ln terms)
    output_Data['scatter'] = np.median(scatter, axis=0) * Ln10
    output_Data['scatter-'] = np.percentile(scatter, percentile[0], axis=0) * Ln10
    output_Data['scatter+'] = np.percentile(scatter, percentile[1], axis=0) * Ln10

    ax[1].set_xlabel(labels[0], size=fontsize.xlabel)
    ax[0].set_ylabel(r"$\alpha\,$(%s)" % labels[1], size=fontsize.ylabel)
    ax[1].set_ylabel(r"$\sigma\,$(%s)" % labels[1], size=fontsize.ylabel)

    return output_Data, ax

def Plot_Fit_Summary(df, xlabel, ylabel, nbins=25, xrange=None, nBootstrap=100, show_data=False,
                    kernel_type='gaussian', kernel_width=0.2, percentile=[16., 84.], ls = '-',
                    xlog=False, ylog = False, labels=None, color=None, verbose=True, ax=None):

    lm = kllr_model(kernel_type, kernel_width)

    if ax is None:
        fig, ax = plt.subplots(3, figsize=(12, 19), sharex=True,
                               gridspec_kw={'height_ratios': [1.75, 1, 1]})
        fig.subplots_adjust(hspace=0.05)

    # Set x_scale to log. Leave y_scale as is.
    if xlog:
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[2].set_xscale('log')

    if ylog:
        ax[0].set_yscale('log')

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()

    if labels is None:
        labels = [xlabel, ylabel]

    # Dictionary to store output values
    output_Data = {}

    # Load and mask data
    x_data, y_data = np.array(df[xlabel]), np.array(df[ylabel])

    mask = np.invert(np.isinf(x_data)) & np.invert(np.isinf(y_data))

    x_data, y_data = x_data[mask], y_data[mask]

    # Generate new arrays to store params in for each Bootstrap realization
    yline   = np.empty([nBootstrap, nbins])
    scatter = np.empty([nBootstrap, nbins])
    slope   = np.empty([nBootstrap, nbins])
    intercept = np.empty([nBootstrap, nbins])

    if verbose:
        iterations_list = tqdm(range(nBootstrap))
    else:
        iterations_list = range(nBootstrap)

    for iBoot in iterations_list:

        # First bootstrap realization is always just raw data
        if iBoot == 0:
            xx, yy = x_data, y_data
        # All other bootstraps have shuffled data
        else:
            xx, index = lm.subsample(x_data)
            yy = y_data[index]

        # xline is always the same regardless of bootstrap so don't need 2D array for it.
        # yline is not needed for plotting in this module so it's a 'dummy' variable
        xline, yline[iBoot, :], intercept[iBoot, :], slope[iBoot, :], scatter[iBoot, :] = lm.fit(xx, yy,
                                                                                                 xrange=xrange,
                                                                                                 nbins=nbins)[:5]

    if xlog: xline = 10 ** xline
    if ylog: yline = 10 ** yline

    # Add black line around regular line to improve visibility
    ax[0].plot(xline, np.e**np.median(np.log(yline), axis = 0), lw=6, color = 'k', label="", ls = ls)
    p = ax[0].plot(xline, np.e**np.median(np.log(yline), axis = 0), lw=3, color = color, ls = ls)
    color = p[0].get_color()

    if show_data:

        #Generate Mask so raw data is shown only for values of x_data within xrange
        if xrange == None:
            #If no xrange inputted, mask selected all available data
            Mask = np.ones(x_data.size, bool)
        else:
            Mask = (x_data > xrange[0]) & (x_data < xrange[1])

        x_data, y_data = x_data[Mask], y_data[Mask]

        if xlog: x_data = 10 ** x_data
        if ylog: y_data = 10 ** y_data
        ax[0].scatter(x_data, y_data, s=30, alpha=0.3, color = color, label="")

    ax[0].set_ylabel(labels[1], size=fontsize.ylabel)

    ax[1].plot(xline, np.median(slope, axis=0), lw=3, color=color, ls = ls)
    ax[1].fill_between(xline, np.percentile(slope, percentile[0], axis=0), np.percentile(slope, percentile[1], axis=0),
                       alpha=0.4, label=None, color=color)

    if ylog:
        ax[2].plot(xline, np.median(scatter, axis=0) * Ln10, lw=3, color=color, ls = ls)
        ax[2].fill_between(xline, np.percentile(scatter, percentile[0], axis=0) * Ln10, np.percentile(scatter, percentile[1], axis=0) * Ln10,
                           alpha=0.4, label=None, color=color)
    else:
        ax[2].plot(xline, np.median(scatter, axis=0), lw=3, color=color, ls = ls)
        ax[2].fill_between(xline, np.percentile(scatter, percentile[0], axis=0), np.percentile(scatter, percentile[1], axis=0),
                           alpha=0.4, label=None, color=color)

    ax[2].set_xlabel(labels[0], size=fontsize.xlabel)
    ax[1].set_ylabel(r"$\alpha\,$(%s)" % labels[1], size=fontsize.ylabel)
    ax[2].set_ylabel(r"$\sigma\,$(%s)" % labels[1], size=fontsize.ylabel)

    output_Data['x']  = xline

    if ylog:
        output_Data['y']  = np.e**np.median(np.log(yline), axis = 0)
        output_Data['y-'] = np.e**np.percentile(np.log(yline), percentile[0], axis=0)
        output_Data['y+'] = np.e**np.percentile(np.log(yline), percentile[1], axis=0)
    else:
        output_Data['y']  = np.median(yline, axis = 0)
        output_Data['y-'] = np.percentile(yline, percentile[0], axis=0)
        output_Data['y+'] = np.percentile(yline, percentile[1], axis=0)

    output_Data['slope']  = np.median(slope, axis=0)
    output_Data['slope-'] = np.percentile(slope, percentile[0], axis=0)
    output_Data['slope+'] = np.percentile(slope, percentile[1], axis=0)

    # Output data for scatter (in ln terms)
    output_Data['scatter']  = np.median(scatter, axis=0) * Ln10
    output_Data['scatter-'] = np.percentile(scatter, percentile[0], axis=0) * Ln10
    output_Data['scatter+'] = np.percentile(scatter, percentile[1], axis=0) * Ln10

    return output_Data, ax

def Plot_Multivariate_Fit_Params(df, xlabels, ylabel, nbins=25, xrange=None, nBootstrap=100, show_all_slopes = False,
                                 kernel_type='gaussian', kernel_width=0.2, percentile=[16., 84.],
                                 xlog=False, labels=None, color=None, verbose=True, ax=None):

    lm = kllr_model(kernel_type, kernel_width)

    if ax is None:
        fig, ax = plt.subplots(2, figsize=(12, 10), sharex=True)
        fig.subplots_adjust(hspace=0.05)

    # Set x_scale to log. Leave y_scale as is.
    if xlog:
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')

    ax[0].grid()
    ax[1].grid()

    if labels is None:
        labels = [xlabels[0], ylabel]

    # Dictionary to store output values
    output_Data = {}

    # Load and mask data
    X_data, y_data = df[xlabels].to_numpy(), np.array(df[ylabel])

    # mask = np.invert(np.isinf(x_data)) & np.invert(np.isinf(y_data))
    #
    # x_data, y_data = x_data[mask], y_data[mask]

    # Generate new arrays to store params in for each Bootstrap realization
    scatter = np.empty([nBootstrap, nbins])
    slope = np.empty([nBootstrap, nbins, X_data.shape[1]])

    if verbose:
        iterations_list = tqdm(range(nBootstrap))
    else:
        iterations_list = range(nBootstrap)

    for iBoot in iterations_list:

        # First bootstrap realization is always just raw data
        if iBoot == 0:
            xx, yy = X_data, y_data
        # All other bootstraps have shuffled data
        else:
            index = lm.subsample(X_data[:, 0])[1]
            xx = X_data[index, :]
            yy = y_data[index]

        # xline is always the same regardless of bootstrap so don't need 2D array for it.
        # yline is not needed for plotting in this module so it's a 'dummy' variable
        xline, slope[iBoot, :, :], scatter[iBoot, :], _ = lm.multivariate_fit(xx, yy, xrange=xrange, nbins=nbins)

    if xlog: xline = 10 ** xline

    for i in range(X_data.shape[1]):

        if (not show_all_slopes) and (i > 0): break
        p = ax[0].plot(xline, np.mean(slope[:, :, i], axis=0), lw=3, color=color,
                       alpha = 1 - i/X_data.shape[1]*0.8 if show_all_slopes else 1)
        color = p[0].get_color()
        ax[0].fill_between(xline, np.percentile(slope[:, :, i], percentile[0], axis=0),
                           np.percentile(slope[:, :, i], percentile[1], axis=0),
                           alpha=0.4, label=None, color=color)
    ax[1].plot(xline, np.mean(scatter, axis=0) * Ln10, lw=3, color=color)
    ax[1].fill_between(xline, np.percentile(scatter, percentile[0], axis=0) * Ln10, np.percentile(scatter, percentile[1], axis=0) * Ln10,
                       alpha=0.4, label=None, color=color)

    # Output Data
    output_Data['x'] = xline

    for i in range(X_data.shape[1]):
        output_Data['slope' + str(i)] = np.median(slope[:, :, i], axis=0)
        output_Data['slope' + str(i) + '-'] = np.percentile(slope[:, :, i], percentile[0], axis=0)
        output_Data['slope' + str(i) + '+'] = np.percentile(slope[:, :, i], percentile[1], axis=0)

    # Output data for scatter (in ln terms)
    output_Data['scatter'] = np.median(scatter, axis=0) * Ln10
    output_Data['scatter-'] = np.percentile(scatter, percentile[0], axis=0) * Ln10
    output_Data['scatter+'] = np.percentile(scatter, percentile[1], axis=0) * Ln10

    ax[1].set_xlabel(labels[0], size=fontsize.xlabel)
    ax[0].set_ylabel(r"$\alpha\,$(%s)" % labels[1], size=fontsize.ylabel)
    ax[1].set_ylabel(r"$\sigma\,$(%s)" % labels[1], size=fontsize.ylabel)

    return output_Data, ax

def Plot_Fit_Params_Split(df, xlabel, ylabel, split_label, split_bins=[], split_mode='Data', nbins=25,
                          xrange=None, nBootstrap=100, kernel_type='gaussian', kernel_width=0.2,
                          xlog=False, percentile=[16., 84.], color=None, labels=None, verbose=True, ax=None):

    check_attributes(split_bins=split_bins, split_mode=split_mode)

    lm = kllr_model(kernel_type, kernel_width)

    if ax is None:
        fig, ax = plt.subplots(2, figsize=(12, 10), sharex=True)
        fig.subplots_adjust(hspace=0.05)

    color = setup_color(color, split_bins, cmap=None)

    ax[0].grid()
    ax[1].grid()

    # Set x_scale to log. Leave y_scale as is.
    if xlog:
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')

    if labels is None:
        labels = [xlabel, ylabel, split_label]

    # Load data and mask it
    x_data, y_data, split_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[split_label])

    mask = np.invert(np.isinf(x_data)) & np.invert(np.isinf(y_data)) & np.invert(np.isinf(split_data))

    x_data, y_data, split_data = x_data[mask], y_data[mask], split_data[mask]

    # Choose bin edges for binning data
    if (isinstance(split_bins, int)):
        if split_mode == 'Data':
            split_bins = [np.percentile(split_data, float(i / split_bins) * 100) for i in range(0, split_bins + 1)]
        elif split_mode == 'Residuals':
            split_res = lm.calculate_residual(x_data, split_data, xrange=xrange)
            split_bins = [np.percentile(split_res, float(i / split_bins) * 100) for i in range(0, split_bins + 1)]

            split_res = lm.calculate_residual(x_data, split_data)

    # Need to compute residuals if split_mode == 'Residuals' is chosen
    elif isinstance(split_bins, (np.ndarray, list, tuple)) & (split_mode == 'Residuals'):
        split_res = lm.calculate_residual(x_data, split_data)

    # Define Output_Data variable to store all computed data that is then plotted
    output_Data = {'Bin' + str(i): {} for i in range(len(split_bins) - 1)}

    for i in range(len(split_bins) - 1):

        if split_mode == 'Data':
            split_mask = (split_data <= split_bins[i + 1]) & (split_data > split_bins[i])
        elif split_mode == 'Residuals':
            split_mask = (split_res <= split_bins[i + 1]) & (split_res > split_bins[i])

        scatter = np.empty([nBootstrap, nbins])
        slope = np.empty([nBootstrap, nbins])
        intercept = np.empty([nBootstrap, nbins])

        sorted_array = np.sort(x_data[split_mask])
        xrange_temp = [np.max([sorted_array[20], xrange[0]]),
                       np.min([sorted_array[-20], xrange[1]])]

        if verbose:
            iterations_list = tqdm(range(nBootstrap))
        else:
            iterations_list = range(nBootstrap)

        for iBoot in iterations_list:

            # First bootstrap realization is always just raw data
            if iBoot == 0:
                xx, yy = x_data[split_mask], y_data[split_mask]
            # All other bootstraps have shuffled data
            else:
                xx, index = lm.subsample(x_data[split_mask])
                yy = y_data[split_mask][index]

            xline, yline, intercept[iBoot, :], \
            slope[iBoot, :], scatter[iBoot, :] = lm.fit(xx, yy, xrange=xrange_temp, nbins=nbins)[:5]

        if split_mode == 'Data':
            label = r'$%0.2f <$ %s $< %0.2f$' % (split_bins[i], labels[2], split_bins[i + 1])
        elif split_mode == 'Residuals':
            label = r'$%0.2f < {\rm res}($%s$) < %0.2f$' % (split_bins[i], labels[2], split_bins[i + 1])

        if xlog: xline = 10 ** xline

        ax[0].plot(xline, np.median(slope, axis=0), lw=3, label=label, color=color[i])
        ax[0].fill_between(xline, np.percentile(slope, percentile[0], axis=0), np.percentile(slope, percentile[1], axis=0),
                           alpha=0.4, label=None, color=color[i])

        # Divide scatter by log10(e) to get it in ln terms (not log10 terms)
        ax[1].plot(xline, np.median(scatter, axis=0) * Ln10, lw=3, label=label, color=color[i])
        ax[1].fill_between(xline,
                           np.percentile(scatter, percentile[0], axis=0) * Ln10,
                           np.percentile(scatter, percentile[1], axis=0) * Ln10,
                           alpha=0.4, label=None, color=color[i])

        # Output xvals
        output_Data['Bin' + str(i)]['x'] = xline

        # Output data for slope
        output_Data['Bin' + str(i)]['slope'] = np.median(slope, axis=0)
        output_Data['Bin' + str(i)]['slope-'] = np.percentile(slope, percentile[0], axis=0)
        output_Data['Bin' + str(i)]['slope+'] = np.percentile(slope, percentile[1], axis=0)

        # Output data for scatter (in ln terms)
        output_Data['Bin' + str(i)]['scatter'] = np.median(scatter, axis=0) * Ln10
        output_Data['Bin' + str(i)]['scatter-'] = np.percentile(scatter, percentile[0], axis=0) * Ln10
        output_Data['Bin' + str(i)]['scatter+'] = np.percentile(scatter, percentile[1], axis=0) * Ln10

    ax[1].set_xlabel(labels[0], size=fontsize.xlabel)
    ax[0].set_ylabel(r"$\alpha\,$(%s)" % labels[1], size=fontsize.ylabel)
    ax[1].set_ylabel(r"$\sigma\,$(%s)" % labels[1], size=fontsize.ylabel)
    ax[1].legend(fontsize=fontsize.legend)

    return output_Data, ax

def Plot_Cov_Corr(df, xlabel, ylabel, zlabel, nbins=25, xrange=None, nBootstrap=100,
                  Output_mode='Covariance', kernel_type='gaussian', kernel_width=0.2,
                  percentile=[16., 84.], xlog=False, labels=None, color=None,
                  verbose=True, ax=None):

    check_attributes(Output_mode=Output_mode)

    lm = kllr_model(kernel_type, kernel_width)

    #Dictionary to store values
    output_Data = {}

    if ax == None:
        ax = plt.figure(figsize = (12,8))

    if (labels is None) or (len(labels) < 3):
        labels = [xlabel, ylabel, zlabel]

    #Load x-vals just so we can set xrange and xline

    if xrange == None:
        x_data = np.array(df[xlabel])
        xrange = (np.min(x) - 0.01, np.max(x) + 0.01)
    elif xrange[0] == None:
        x_data    = np.array(df[xlabel])
        xrange[0] = np.min(x) - 0.01
    elif xrange[1] == None:
        x_data    = np.array(df[xlabel])
        xrange[1] = np.max(x) + 0.01

    xline = np.linspace(xrange[0], xrange[1], nbins, endpoint=True)

    if verbose:
        iterations_list = tqdm(range(len(xline)))
    else:
        iterations_list = range(len(xline))

    if kernel_width is None:
        kernel_width = self.kernel_width

    #If kernel_width is just one number, then turn it
    #into an array of length = nbins
    if isinstance(kernel_width, (int, float)):
        kernel_width_array = np.ones(nbins)*kernel_width

    elif len(kernel_width) != len(xline):
        #Check if kernel_width has right dimensions
        raise ValueError("Size mismatch: kernel_width has length %d. Paramater nbins = %d"%(len(kernel_width), nbins))
    #Otherwise rename value for rest of code
    else:
        kernel_width_array = kernel_width

    x_data, y_data, z_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[zlabel])
    mask = np.invert(np.isinf(x_data)) & (np.invert(np.isinf(y_data))) & (np.invert(np.isinf(z_data)))
    x_data, y_data, z_data = x_data[mask], y_data[mask], z_data[mask]

    cov_corr = np.zeros([nBootstrap, len(xline)])

    for k in iterations_list:

        #Add mask to reduce data and improve
        #efficiency (hopefully)
        x_Mask = (x_data > xline[k] - 3*kernel_width_array[k]) & (x_data < xline[k] + 3*kernel_width_array[k])

        x_masked, y_masked, z_masked = x_data[x_Mask], y_data[x_Mask], z_data[x_Mask]

        if Output_mode.lower() in ['covariance', 'cov']:
            cov_corr[:, k] = lm.calc_covariance_fixed_x(x_masked, y_masked, z_masked, xline[k],
                                                        kernel_width=kernel_width_array[k],
                                                        parametric_bootstrap = True,
                                                        nBootstrap = nBootstrap)
        elif Output_mode.lower() in ['correlation', 'corr']:
            cov_corr[:, k] = lm.calc_correlation_fixed_x(x_masked, y_masked, z_masked, xline[k],
                                                         kernel_width=kernel_width_array[k],
                                                         parametric_bootstrap = True,
                                                         nBootstrap = nBootstrap)

    output_Data['x'] = xline

    if Output_mode.lower() in ['covariance', 'cov']:
        name = 'cov'
    elif Output_mode.lower() in ['correlation', 'corr']:
        name = 'corr'

    output_Data['%s_%s_%s'%(name, ylabel, zlabel)]  = np.median(cov_corr, axis=0)
    output_Data['%s_%s_%s-'%(name, ylabel, zlabel)] = np.percentile(cov_corr, percentile[0], axis=0)
    output_Data['%s_%s_%s+'%(name, ylabel, zlabel)] = np.percentile(cov_corr, percentile[1], axis=0)

    if xlog: plt.xscale('log')

    if xlog:
        xline_temp = 10 ** (xline)
    else:
        xline_temp = xline

    p = plt.plot(xline_temp, np.mean(cov_corr, axis=0), lw=3, color=color)
    color = p[0].get_color()
    plt.fill_between(xline_temp, np.percentile(cov_corr, percentile[0], axis=0),
                        np.percentile(cov_corr, percentile[1], axis=0), alpha=0.4, label=None, color=color)
    plt.grid()

    if Output_mode.lower() in ['correlation', 'corr']:
        plt.axhline(y=0.0, color='k', lw=2)
        plt.ylim(ymin=-1, ymax=1)

    if Output_mode.lower() in ['correlation', 'corr']:
        plt.ylabel(r'r(%s-%s)'%(labels[1], labels[2]), size=fontsize.ylabel)
    else:
        plt.ylabel(r'cov(%s-%s)'%(labels[1], labels[2]), size=fontsize.ylabel)

    plt.xlabel(labels[0], size=fontsize.xlabel)

    return output_Data, ax

def Plot_Cov_Corr_Matrix(df, xlabel, ylabels, nbins=25, xrange=None, nBootstrap=100,
                         Output_mode='Covariance', kernel_type='gaussian', kernel_width=0.2,
                         percentile=[16., 84.], xlog=False, labels=None, color=None,
                         verbose=True, ax=None, c200c_offset = 1, ls = '-'):

    check_attributes(Output_mode=Output_mode)

    from scipy import interpolate

    def N_Sigma_Filter(x_data, y_data, N_sigma = 5, xrange = None):

        if xrange == None:

            xrange = (np.min(x_data), np.max(x_data))

        model = kllr_model()
        x, y, pi, alpha, eps = model.fit(x_data, y_data, kernel_width=0.2, nbins=50, xrange=xrange)

        scatter  = interpolate.UnivariateSpline(x, eps, k = 1)

        mean_val = interpolate.UnivariateSpline(x, y, k = 1)

        upper_bound = mean_val(x_data) + N_sigma*scatter(x_data)
        lower_bound = mean_val(x_data) - N_sigma*scatter(x_data)

        Mask = (y_data > lower_bound) & (y_data < upper_bound)

        return Mask

    lm = kllr_model(kernel_type, kernel_width)

    #Dictionary to store values
    output_Data = {}

    # size of matrix
    if Output_mode.lower() in ['covariance', 'cov']:

        # 'length' of matrix is same as number of properties
        matrix_size = len(ylabels)

        if ax is None:
            fig = plt.figure(figsize=(5 * matrix_size, 5 * matrix_size))

            # Do not share y-axes, since covariance can have different amplitudes
            ax = fig.subplots(matrix_size, matrix_size, sharex=True, sharey=False)

    elif Output_mode.lower() in ['correlation', 'corr']:

        # 'length' of matrix is one less than number of properties
        matrix_size = len(ylabels) - 1

        if ax is None:
            fig = plt.figure(figsize=(5 * matrix_size, 5 * matrix_size))

            # Share y-axes since by definition, correlation must be within -1 <= r <= 1
            ax = fig.subplots(matrix_size, matrix_size, sharex=True, sharey=True)

    if matrix_size > 1:
        for i in range(matrix_size):
            for j in range(matrix_size):
                ax[i, j].axis('off')

    if labels is None:
        ylabels.sort()
        labels = [xlabel] + ylabels
    else:
        None
        # Sort ylabels alphabetically but make sure we also sort the label list (if provided) in sync
        # ylabels, temp = zip(*sorted(zip(ylabels, labels[1:])))
        # ylabels, labels[1:] = list(ylabels), list(temp)

    col = -1

    if verbose:
        iterations_list = tqdm(ylabels)
    else:
        iterations_list = ylabels

    #Load x-vals just so we can set xrange and xline

    if xrange == None:
        x_data = np.array(df[xlabel])
        xrange = (np.min(x) - 0.01, np.max(x) + 0.01)
    elif xrange[0] == None:
        x_data    = np.array(df[xlabel])
        xrange[0] = np.min(x) - 0.01
    elif xrange[1] == None:
        x_data    = np.array(df[xlabel])
        xrange[1] = np.max(x) + 0.01

    xline_true = np.linspace(xrange[0], xrange[1], nbins, endpoint=True)

    if kernel_width is None:
        kernel_width = self.kernel_width

    #If kernel_width is just one number, then turn it
    #into an array of length = nbins
    if isinstance(kernel_width, (int, float)):
        kernel_width_array = np.ones(nbins)*kernel_width

    elif len(kernel_width) != len(xline_true):
        #Check if kernel_width has right dimensions
        raise ValueError("Size mismatch: kernel_width has length %d. Paramater nbins = %d"%(len(kernel_width), nbins))
    #Otherwise rename value for rest of code
    else:
        kernel_width_array = kernel_width

    for ylabel in iterations_list:

        col += 1
        row = col

        for zlabel in ylabels:

            i, j = ylabels.index(ylabel), ylabels.index(zlabel)

            if Output_mode.lower() in ['covariance', 'cov']:
                if j < i:
                    continue

            elif Output_mode.lower() in ['correlation', 'corr']:
                if j <= i:
                    continue

            with np.errstate(invalid = 'ignore'):
                x_data, y_data, z_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[zlabel])
                mask = ((x_data > -1e10) & (x_data < 1e10) &
                        (y_data > -1e10) & (y_data < 1e10) &
                        (z_data > -1e10) & (z_data < 1e10))
                x_data, y_data, z_data = x_data[mask], y_data[mask], z_data[mask]

            #Perform filter
            Mask_y = N_Sigma_Filter(x_data, y_data, 3,
                                    xrange = [xrange[0] + c200c_offset, xrange[1]] if 'c200c' in ylabel else None)
            Mask_z = N_Sigma_Filter(x_data, z_data, 3,
                                    xrange = [xrange[0] + c200c_offset, xrange[1]] if 'c200c' in zlabel else None)

            if ('c200c' not in ylabel) & ('E_s_DMO' not in ylabel): Mask_y = np.ones(Mask_y.size).astype(bool)
            if ('c200c' not in zlabel) & ('E_s_DMO' not in ylabel): Mask_z = np.ones(Mask_z.size).astype(bool)

            x_data, y_data, z_data = x_data[Mask_y & Mask_z], y_data[Mask_y & Mask_z], z_data[Mask_y & Mask_z]

            if ('c200c' in ylabel) or ('c200c' in zlabel):
                xline = np.linspace(xrange[0] + c200c_offset, xrange[1], nbins, endpoint=True)

                from scipy import interpolate
                kernel_here = interpolate.interp1d(np.linspace(xrange[0], xrange[1], nbins, endpoint=True),
                                                   kernel_width_array)(xline)

            else:
                xline = xline_true
                kernel_here = kernel_width_array
            cov_corr = np.zeros([nBootstrap, len(xline)])

            for k in range(len(xline)):

                #Add mask to reduce data and improve
                #efficiency (hopefully)
                x_Mask = (x_data > xline[k] - 3*kernel_here[k]) & (x_data < xline[k] + 3*kernel_here[k])

                x_masked, y_masked, z_masked = x_data[x_Mask], y_data[x_Mask], z_data[x_Mask]

                if Output_mode.lower() in ['covariance', 'cov']:
                    cov_corr[:, k] = lm.calc_covariance_fixed_x(x_masked, y_masked, z_masked, xline[k],
                                                                kernel_width=kernel_here[k],
                                                                parametric_bootstrap = True,
                                                                nBootstrap = nBootstrap)
                elif Output_mode.lower() in ['correlation', 'corr']:
                    cov_corr[:, k] = lm.calc_correlation_fixed_x(x_masked, y_masked, z_masked, xline[k],
                                                                 kernel_width=kernel_here[k],
                                                                 parametric_bootstrap = True,
                                                                 nBootstrap = nBootstrap)

            output_Data['x'] = xline

            if Output_mode.lower() in ['covariance', 'cov']:
                name = 'cov'
            elif Output_mode.lower() in ['correlation', 'corr']:
                name = 'corr'

            output_Data['%s_%s_%s'%(name, ylabel, zlabel)]  = np.median(cov_corr, axis=0)
            output_Data['%s_%s_%s-'%(name, ylabel, zlabel)] = np.percentile(cov_corr, percentile[0], axis=0)
            output_Data['%s_%s_%s+'%(name, ylabel, zlabel)] = np.percentile(cov_corr, percentile[1], axis=0)

            if matrix_size > 1:
                ax_tmp = ax[row, col]
            else:
                ax_tmp = ax

            if xlog: ax_tmp.set_xscale('log')
            ax_tmp.axis('on')

            if xlog:
                xline_temp = 10 ** (xline)
            else:
                xline_temp = xline

            p = ax_tmp.plot(xline_temp, np.mean(cov_corr, axis=0), lw=3, color=color, ls = ls)
            color = p[0].get_color()
            ax_tmp.fill_between(xline_temp, np.percentile(cov_corr, percentile[0], axis=0),
                                np.percentile(cov_corr, percentile[1], axis=0), alpha=0.4, label=None, color=color)
            ax_tmp.grid()

            if Output_mode.lower() in ['correlation', 'corr']:
                ax_tmp.axhline(y=0.0, color='k', lw=2)
                ax_tmp.set_ylim(ymin=-1.05, ymax=1.05)

            if col == row:

                #Remove any text that exists already
                for text in ax_tmp.texts:
                    text.set_visible(False)

                ax_tmp.text(1.02, 0.5, labels[1 + j], size=fontsize.ylabel,
                            verticalalignment='center', rotation=270,
                            transform=ax_tmp.transAxes)

            if row == col:
                ax_tmp.set_title(labels[1 + i], size=fontsize.xlabel)

            if row == matrix_size - 1:
                ax_tmp.set_xlabel(labels[0], size=fontsize.xlabel)

            if col == 0:
                if Output_mode.lower() in ['correlation', 'corr']:
                    ax_tmp.set_ylabel('r', size=fontsize.xlabel)
                else:
                    ax_tmp.set_ylabel('cov', size=fontsize.xlabel)

            ax_tmp.tick_params(axis='both', which='major', labelsize=12)

            row += 1

    if Output_mode.lower() in ['correlation', 'corr']: plt.subplots_adjust(hspace=0.04, wspace=0.04)
    else: plt.subplots_adjust(hspace=0.04)

    return output_Data, ax

def Plot_Cov_Corr_Matrix_Split(df, xlabel, ylabels, split_label, split_bins=[], Output_mode='Covariance',
                               split_mode='Data', nbins=25, xrange=None, nBootstrap=100,
                               kernel_type='gaussian', kernel_width=0.2, xlog=False, percentile=[16., 84.],
                               labels=None, color=None, verbose=True, ax=None):

    check_attributes(split_bins=split_bins, Output_mode=Output_mode, split_mode=split_mode)

    lm = kllr_model(kernel_type, kernel_width)

    # size of matrix
    if Output_mode.lower() in ['covariance', 'cov']:

        # 'length' of matrix is same as number of properties
        matrix_size = len(ylabels)

        if ax == None:
            fig = plt.figure(figsize=(5 * matrix_size, 5 * matrix_size))

            # Do not share y-axes, since covariance can have different amplitudes
            ax = fig.subplots(matrix_size, matrix_size, sharex=True, sharey=False)

    elif Output_mode.lower() in ['correlation', 'corr']:

        # 'length' of matrix is one less than number of properties
        matrix_size = len(ylabels) - 1

        if ax == None:
            fig = plt.figure(figsize=(5 * matrix_size, 5 * matrix_size))

            # Share y-axes since by definition, correlation must be within -1 <= r <= 1
            ax = fig.subplots(matrix_size, matrix_size, sharex=True, sharey=True)

    color = setup_color(color, split_bins, cmap=None)

    # Set all axes off by default. We will turn on only the lower-left-triangle
    if matrix_size > 1:
        for i in range(matrix_size):
            for j in range(matrix_size):
                ax[i, j].axis('off')

    # if len(labels) < (len(ylabels) + 2):
    if labels is None:
        ylabels.sort()
        labels = [xlabel] + ylabels + [split_label]
    else:
        # Sort ylabels alphebetically but make sure we also sort the label list (if provided) in sync
        ylabels, temp = zip(*sorted(zip(ylabels, labels[1:-1])))
        ylabels, labels[1:-1] = list(ylabels), list(temp)

    # Value to keep track of which column number we're in (leftmost is col = 0)
    # Set it to -1 here so in the first loop it goes to col = 0
    col = -1

    if verbose:
        iterations_list = tqdm(ylabels)
    else:
        iterations_list = ylabels

    for ylabel in iterations_list:

        col += 1
        # Start from the plot on the diagonal
        row = col

        for zlabel in ylabels:

            i, j = ylabels.index(ylabel), ylabels.index(zlabel)

            # Create condition that prevents double-computing the same correlations (eg. corr(x,y) = corr(y,x))
            if Output_mode.lower() in ['covariance', 'cov']:
                if j < i:
                    continue

            elif Output_mode.lower() in ['correlation', 'corr']:
                if j <= i:
                    continue

            if matrix_size > 1:
                ax_tmp = ax[row, col]
            else:
                ax_tmp = ax

            x_data, y_data, z_data, split_data = np.array(df[xlabel]), np.array(df[ylabel]), \
                                                 np.array(df[zlabel]), np.array(df[split_label])

            if xrange is None:
                xrange = [np.min(x_data) - 0.001, np.max(x_data) + 0.001]

            mask = np.invert(np.isinf(x_data)) & np.invert(np.isinf(y_data)) & \
                   np.invert(np.isinf(z_data)) & np.invert(np.isinf(split_data))

            x_data, y_data, z_data, split_data = x_data[mask], y_data[mask], z_data[mask], split_data[mask]

            # Choose bin edges for binning data
            if (isinstance(split_bins, int)):
                if split_mode == 'Data':
                    split_bins = [np.percentile(split_data, float(i / split_bins) * 100) for i in
                                  range(0, split_bins + 1)]
                elif split_mode == 'Residuals':
                    split_res = lm.calculate_residual(x_data, split_data, xrange=xrange)
                    split_bins = [np.percentile(split_res, float(i / split_bins) * 100) for i in
                                  range(0, split_bins + 1)]
            elif isinstance(split_bins, (np.ndarray, list, tuple)) & (split_mode == 'Residuals'):
                split_res = lm.calculate_residual(x_data, split_data, xrange=xrange)

            # Normally, we would define a dictionary for output here
            # However, there is too much data here to print out all data shown in a matrix
            # Instead one can obtain correlation plotting data using just the non-matrix version

            for k in range(len(split_bins) - 1):

                if split_mode == 'Data':
                    split_mask = (split_data <= split_bins[k + 1]) & (split_data > split_bins[k])
                elif split_mode == 'Residuals':
                    split_mask = (split_res < split_bins[k + 1]) & (split_res > split_bins[k])

                xline = np.linspace(xrange[0], xrange[1], nbins)
                # xline = (xline[1:] + xline[:-1]) / 2.

                cov_corr = np.zeros([nBootstrap, len(xline)])

                for iBoot in range(nBootstrap):

                    # First bootstrap realization is always just raw data
                    if iBoot == 0:
                        xx, yy, zz = x_data[split_mask], y_data[split_mask], z_data[split_mask]

                    # All other bootstraps have shuffled data
                    else:
                        xx, index = lm.subsample(x_data[split_mask])
                        yy = y_data[split_mask][index]
                        zz = z_data[split_mask][index]

                    for l in range(len(xline)):

                        if Output_mode.lower() in ['covariance', 'cov']:
                            cov_corr[iBoot, l] = lm.calc_covariance_fixed_x(xx, yy, zz, xline[l])
                        elif Output_mode.lower() in ['correlation', 'corr']:
                            cov_corr[iBoot, l] = lm.calc_correlation_fixed_x(xx, yy, zz, xline[l])

                if split_mode == 'Data':
                    label = r'$%0.2f <$ %s $< %0.2f$' % (split_bins[k], labels[-1], split_bins[k + 1])
                elif split_mode == 'Residuals':
                    label = r'$%0.2f < {\rm res}($%s$) < %0.2f$' % (split_bins[k], labels[-1], split_bins[k + 1])

                if xlog:
                    xline = 10 ** (xline)
                    ax_tmp.set_xscale('log')

                ax_tmp.axis('on')
                ax_tmp.plot(xline, np.mean(cov_corr, axis=0), lw=3, color=color[k], label=label)
                ax_tmp.fill_between(xline,
                                    np.percentile(cov_corr, percentile[0], axis=0),
                                    np.percentile(cov_corr, percentile[1], axis=0),
                                    alpha=0.4, label=None, color=color[k])

            ax_tmp.grid()

            if Output_mode.lower() in ['correlation', 'corr']:
                ax_tmp.axhline(y=0.0, color='k', lw=2)
                ax_tmp.set_ylim(ymin=-1, ymax=1)

            if col == row:
                ax_tmp.text(1.02, 0.5, labels[1 + j], size=fontsize.ylabel,
                            horizontalalignment='left', verticalalignment='center', rotation=270, clip_on=False,
                            transform=ax_tmp.transAxes)
            if row == col:
                ax_tmp.set_title(labels[1 + i], size=fontsize.xlabel)

            if row == matrix_size - 1:
                ax_tmp.set_xlabel(labels[0], size=fontsize.xlabel)

            if col == 0:
                if Output_mode.lower() in ['correlation', 'corr']:
                    ax_tmp.set_ylabel('r', size=fontsize.xlabel)
                else:
                    ax_tmp.set_ylabel('cov', size=fontsize.xlabel)

            ax_tmp.tick_params(axis='both', which='major', labelsize=12)

            row += 1

    if matrix_size > 1:
        if matrix_size % 2 == 1:
            ax[matrix_size // 2, matrix_size // 2].legend(prop={'size': 8 + 4 * matrix_size}, loc=(1.1, 1.3))
        else:
            ax[matrix_size // 2, matrix_size // 2].legend(prop={'size': 8 + 4 * matrix_size}, loc=(0.4, 1.9))

        legend = ax[matrix_size // 2, matrix_size // 2].get_legend()
        for i in range(len(split_bins) - 1):
            legend.legendHandles[i].set_linewidth(2 + 0.5 * matrix_size)
        if Output_mode.lower() in ['correlation', 'corr']: plt.subplots_adjust(hspace=0.04, wspace=0.04)
        else: plt.subplots_adjust(hspace=0.04)
    else:
        plt.legend(fontsize=fontsize.legend)

    return ax

def Plot_Residual(df, xlabel, ylabel, nbins=15, xrange=None, PDFrange=(-4, 4), nBootstrap=1000,
                  kernel_type='gaussian', kernel_width=0.2, percentile=[16., 84.],
                  funcs={}, labels=None, color=None, verbose=True, ax=None):

    lm = kllr_model(kernel_type, kernel_width)

    if ax == None:
        ax = plt.figure(figsize=(12, 8))

    plt.grid()

    if labels is None:
        labels = [r'normalized residuals of %s' % ylabel, 'PDF']
    else:
        labels = [r'normalized residuals of %s' % labels[0], 'PDF', labels[1]]

    # Dictionary that will store values to be output
    output_Data = {}
    results = funcs.keys()

    x_data, y_data = np.array(df[xlabel]), np.array(df[ylabel])

    mask = np.invert(np.isinf(x_data)) & np.invert(np.isinf(y_data) | np.isnan(y_data))

    x_data, y_data = x_data[mask], y_data[mask]

    dy = lm.calculate_residual(x_data, y_data, xrange=xrange, nbins = nbins)

    output_Data['Residuals'] = dy

    PDFs, bins, output = lm.PDF_generator(dy, nbins, nBootstrap, funcs, xrange=PDFrange, density=True, verbose=verbose)

    for r in results:
        min = np.percentile(output[r], percentile[0])
        mean = np.mean(output[r])
        max = np.percentile(output[r], percentile[1])
        print(r, ":", np.round(min - mean, 4), np.round(mean, 4), np.round(max - mean, 4))

        output_Data[r + '-'] = np.percentile(output[r], percentile[0])
        output_Data[r] = np.median(output[r])
        output_Data[r + '+'] = np.percentile(output[r], percentile[1])

    p = plt.plot(bins, np.mean(PDFs, axis=0), lw=3, color=color)
    color = p[0].get_color()
    plt.fill_between(bins, np.percentile(PDFs, percentile[0], axis=0), np.percentile(PDFs, percentile[1], axis=0),
                     alpha=0.4, label=None, color=color)

    plt.xlabel(labels[0], size=fontsize.xlabel)
    plt.ylabel(labels[1], size=fontsize.ylabel)
    plt.legend(fontsize=fontsize.legend)

    return output_Data, ax

def Plot_Residual_Split(df, xlabel, ylabel, split_label, split_bins=[], split_mode='Data', nbins=15, xrange=None,
                        PDFrange=(-4, 4), nBootstrap=1000, kernel_type='gaussian', kernel_width=0.2,
                        percentile=[16., 84.], labels=None, funcs={}, color=None, verbose=True, ax=None):

    check_attributes(split_bins=split_bins, split_mode=split_mode)

    lm = kllr_model(kernel_type, kernel_width)

    if ax == None:
        ax = plt.figure(figsize=(12, 8))

    color = setup_color(color, split_bins, cmap=None)

    plt.grid()

    if labels is None:
        labels = [r'normalized residuals of %s' % ylabel, 'PDF', split_label]
    else:
        labels = [r'normalized residuals of %s' % labels[0], 'PDF', labels[1]]

    # Dictionary that will store values to be output
    results = funcs.keys()

    # Load data and mask it
    x_data, y_data, split_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[split_label])

    mask = np.invert(np.isinf(x_data)) & np.invert(np.isinf(y_data) | np.isnan(y_data)) & np.invert(
        np.isinf(split_data) | np.isnan(split_data))

    x_data, y_data, split_data = x_data[mask], y_data[mask], split_data[mask]

    if xrange is None:
        xrange = [np.min(x_data) - 0.001, np.max(x_data) + 0.001]

    # Choose bin edges for binning data
    if isinstance(split_bins, int):
        if split_mode == 'Data':
            split_bins = [np.percentile(split_data, float(i / split_bins) * 100) for i in range(0, split_bins + 1)]
        elif split_mode == 'Residuals':
            split_res = lm.calculate_residual(x_data, split_data, xrange=xrange)
            split_bins = [np.percentile(split_res, float(i / split_bins) * 100) for i in range(0, split_bins + 1)]
    elif isinstance(split_bins, (np.ndarray, list, tuple)) & (split_mode == 'Residuals'):
        split_res = lm.calculate_residual(x_data, split_data, xrange=xrange)

    # Define Output_Data variable to store all computed data that is then plotted
    output_Data = {'Bin' + str(i): {} for i in range(len(split_bins) - 1)}

    # Compute LLR and Residuals using full dataset
    # We do this so the LLR parameters are shared between the different split_bins
    # And that way differences in the PDF are inherent
    # Modulating LLR params for each split_bin would wash away the differences in
    # the PDFs of each split_bin
    dy = lm.calculate_residual(x_data, y_data, xrange=xrange)

    #Generate mask so only objects within xrange are used in PDF
    #Need this to make sure dy and split_data are the same length
    xrange_mask = (x_data < xrange[1]) & (x_data > xrange[0])

    # Separately plot the PDF of data in each bin
    for i in range(len(split_bins) - 1):

        if split_mode == 'Data':
            split_mask = (split_data[xrange_mask] < split_bins[i + 1]) & (split_data[xrange_mask] > split_bins[i])

        elif split_mode == 'Residuals':
            split_mask = (split_res[xrange_mask] < split_bins[i + 1]) & (split_res[xrange_mask] > split_bins[i])

        output_Data['Bin' + str(i)]['Residuals'] = dy[split_mask]

        PDFs, bins, output = lm.PDF_generator(dy[split_mask], nbins, nBootstrap, funcs,
                                              xrange=PDFrange, density=True, verbose=verbose)

        for r in results:
            min = np.percentile(output[r], percentile[0])
            mean = np.mean(output[r])
            max = np.percentile(output[r], percentile[1])
            print(r, ":", np.round(min - mean, 4), np.round(mean, 4), np.round(max - mean, 4))

            output_Data['Bin' + str(i)][r + '-'] = np.percentile(output[r], percentile[0])
            output_Data['Bin' + str(i)][r] = np.median(output[r])
            output_Data['Bin' + str(i)][r + '+'] = np.percentile(output[r], percentile[1])

        if split_mode == 'Data':
            label = r'$%0.2f <$ %s $< %0.2f$' % (split_bins[i], labels[2], split_bins[i + 1])
        elif split_mode == 'Residuals':
            label = r'$%0.2f < {\rm res}($%s$) < %0.2f$' % (split_bins[i], labels[2], split_bins[i + 1])

        plt.plot(bins, np.mean(PDFs, axis=0), lw=3, color=color[i], label=label)
        plt.fill_between(bins, np.percentile(PDFs, percentile[0], axis=0), np.percentile(PDFs, percentile[1], axis=0),
                         alpha=0.4, label=None, color=color[i])

    plt.xlabel(labels[0], size=fontsize.xlabel)
    plt.ylabel(labels[1], size=fontsize.ylabel)
    plt.legend(fontsize=fontsize.legend)

    return output_Data, ax

def Plot_Shapley_Values(df, xlabel, ylabel, zlabels, nbins=25, xrange=None, nBootstrap=100, n_jobs = 1,
                        kernel_type='gaussian', kernel_width=0.2, percentile=[16., 84.], approximate = True,
                        xlog=False, labels=None, color=None, verbose=True, ax=None, max_samples = None,
                        n_estimators = 100, normalized = False, iter_smooth = 3, show_error = False,
                        no_plot = False):

    lm = kllr_model()

    if ax == None:
        ax = plt.figure(figsize=(12, 8))

    if xlog:
        plt.xscale('log')

    plt.grid()

    if labels is None:
        labels = [xlabel, r'$\mu(|\phi_{\rm SHAP}/\Delta{' + ylabel + '}|)$'] + zlabels
    else:
        labels = [labels[0], r'$\mu(|\phi_{\rm SHAP}/\Delta{' + labels[1] + '}|)$'] + labels[2:]

    Mask = (df != np.inf) & (df != -np.inf) & np.invert(df.isnull())

    df = df[Mask]

    x_data, y_data, z_data = np.array(df[xlabel]), np.array(df[ylabel]), df[zlabels].to_numpy()

    dy = lm.calculate_residual(x_data, y_data)

    dz_data = np.zeros_like(z_data)

    # for z_ind in range(dz_data.shape[1]):
    #
    #     print(z_ind)
    #
    #     dz_data[:, z_ind] = lm.calculate_residual(x_data, z_data[:, z_ind])

    xline, phi_SHAP = lm.Shapley_values(x_data, dy, z_data, xrange, nbins,
                                        kernel_type = kernel_type, kernel_width = kernel_width, n_jobs = n_jobs,
                                        max_samples = max_samples, n_estimators = n_estimators)

    phi_SHAP /= dy[:, np.newaxis]
    phi_SHAP  = np.abs(phi_SHAP)

    Median_contribution = np.nanpercentile(phi_SHAP, 50., axis = 1)
    Upper_contribution  = np.nanpercentile(phi_SHAP, percentile[1], axis = 1)
    Lower_contribution  = np.nanpercentile(phi_SHAP, percentile[0], axis = 1)

    for j in range(iter_smooth):
        Median_contribution[1:-1, :] = (Median_contribution[:-2,  :] +
                                        2*Median_contribution[1:-1, :] +
                                        Median_contribution[2:,   :])/4

    if xlog:
        x_temp = 10**xline
    else:
        x_temp = xline

    # plt.plot(x_temp, np.nanmedian(np.sum(normed_phi_SHAP, axis = 2), axis = 1), lw = 3)

    # print(labels, Median_contribution.shape)
    for i in range(len(zlabels)):
        p = plt.plot(x_temp, Median_contribution[:, i], label = labels[i + 2], lw = 3)
        color = p[0].get_color()
        if show_error:
            plt.fill_between(x_temp, Upper_contribution[:, i], Lower_contribution[:, i], alpha = 0.15, color = color)

    plt.xlabel(labels[0], size = fontsize.xlabel)
    plt.ylabel(labels[1], size = fontsize.ylabel)
    plt.legend(fontsize = fontsize.legend)

    if no_plot: plt.close()

    return phi_SHAP, ax

def check_attributes(split_bins=10, Output_mode='corr', split_mode='Data'):

    if not isinstance(split_bins, int) and not isinstance(split_bins, (np.ndarray, list, tuple)):
        raise TypeError("split_bins must be an integer number or a list of float numbers, "
                        "split_bins is type '%s' "%type(split_bins))
    elif isinstance(split_bins, int) and split_bins < 2:
        raise ValueError('split_bins must be an integer number larger than 1, split_bins is %i'%split_bins)
    elif isinstance(split_bins, int) and split_bins < 2:
        raise ValueError('len(split_bins) must be larger than 1, len(split_bins) is %i'%len(split_bins))

    if Output_mode.lower() not in ['correlation', 'corr', 'covariance', 'cov']:
        raise ValueError("Output_mode must be in ['correlation', 'corr', 'covariance', 'cov']. The passed "
                         "Output_mode is `%s`."%Output_mode)

    if split_mode.lower() not in ['residuals', 'data']:
        raise ValueError("split_mode must be in ['Residuals', 'Data']. The passed "
                         "split_mode is `%s`."%split_mode)



#Remove functions later

def Plot_Fit_Summary_for_Paper(df, xlabel, ylabel, nbins=25, xrange=None, nBootstrap=100, show_data=False,
                               kernel_type='gaussian', kernel_width=0.2, percentile=[16., 84.], ls = '-',
                               xlog=False, ylog = False, labels=None, color=None, verbose=True):

    lm = kllr_model(kernel_type, kernel_width)

    # Dictionary to store output values
    output_Data = {}

    # Load and mask data
    x_data, y_data = np.array(df[xlabel]), np.array(df[ylabel])

    mask = np.invert(np.isinf(x_data)) & np.invert(np.isinf(y_data))

    x_data, y_data = x_data[mask], y_data[mask]

    xline, yline, _, slope, scatter, skew = lm.fit_bootstrapped(x_data, y_data, xrange=xrange, nbins=nbins, nBootstrap = nBootstrap)

    if xlog: xline = 10 ** xline
    if ylog: yline = 10 ** yline

    output_Data['x']  = xline

    if ylog:
        output_Data['y']    = 10**np.median(np.log10(yline), axis = 0)
        output_Data['y-']   = 10**np.percentile(np.log10(yline), 50 - 68/2, axis=0)
        output_Data['y+']   = 10**np.percentile(np.log10(yline), 50 + 68/2, axis=0)
        output_Data['y--']  = 10**np.percentile(np.log10(yline), 50 - 95/2, axis=0)
        output_Data['y++']  = 10**np.percentile(np.log10(yline), 50 + 95/2, axis=0)
        output_Data['y---'] = 10**np.percentile(np.log10(yline), 50 - 99.7/2, axis=0)
        output_Data['y+++'] = 10**np.percentile(np.log10(yline), 50 + 99.7/2, axis=0)
    else:
        output_Data['y']    = np.median(yline, axis = 0)
        output_Data['y-']   = np.percentile(yline, 50 - 68/2, axis=0)
        output_Data['y+']   = np.percentile(yline, 50 + 68/2, axis=0)
        output_Data['y--']  = np.percentile(yline, 50 - 95/2, axis=0)
        output_Data['y++']  = np.percentile(yline, 50 + 95/2, axis=0)
        output_Data['y---'] = np.percentile(yline, 50 - 99.7/2, axis=0)
        output_Data['y+++'] = np.percentile(yline, 50 + 99.7/2, axis=0)

    output_Data['slope']    = np.median(slope, axis=0)
    output_Data['slope-']   = np.percentile(slope, 50 - 68/2, axis=0)
    output_Data['slope+']   = np.percentile(slope, 50 + 68/2, axis=0)
    output_Data['slope--']  = np.percentile(slope, 50 - 95/2, axis=0)
    output_Data['slope++']  = np.percentile(slope, 50 + 95/2, axis=0)
    output_Data['slope---'] = np.percentile(slope, 50 - 99.7/2, axis=0)
    output_Data['slope+++'] = np.percentile(slope, 50 + 99.7/2, axis=0)

    # Output data for scatter (in ln terms)
    output_Data['scatter']    = np.median(scatter, axis=0) * Ln10
    output_Data['scatter-']   = np.percentile(scatter, 50 - 68/2, axis=0) * Ln10
    output_Data['scatter+']   = np.percentile(scatter, 50 + 68/2, axis=0) * Ln10
    output_Data['scatter--']  = np.percentile(scatter, 50 - 95/2, axis=0) * Ln10
    output_Data['scatter++']  = np.percentile(scatter, 50 + 95/2, axis=0) * Ln10
    output_Data['scatter---'] = np.percentile(scatter, 50 - 99.7/2, axis=0) * Ln10
    output_Data['scatter+++'] = np.percentile(scatter, 50 + 99.7/2, axis=0) * Ln10

    # Output for skew
    output_Data['skew']    = np.median(skew, axis=0)
    output_Data['skew-']   = np.percentile(skew, 50 - 68/2, axis=0)
    output_Data['skew+']   = np.percentile(skew, 50 + 68/2, axis=0)
    output_Data['skew--']  = np.percentile(skew, 50 - 95/2, axis=0)
    output_Data['skew++']  = np.percentile(skew, 50 + 95/2, axis=0)
    output_Data['skew---'] = np.percentile(skew, 50 - 99.7/2, axis=0)
    output_Data['skew+++'] = np.percentile(skew, 50 + 99.7/2, axis=0)

    return output_Data

def Plot_Cov_Corr_For_Paper(df, xlabel, ylabel, zlabel, nbins=25, xrange=None, nBootstrap=100,
                            Output_mode='Covariance', kernel_type='gaussian', kernel_width=0.2,
                            percentile=[16., 84.], xlog=False, labels=None, color=None,
                            verbose=True, ax=None):

    check_attributes(Output_mode=Output_mode)

    lm = kllr_model(kernel_type, kernel_width)

    #Dictionary to store values
    output_Data = {}

    if xrange == None:
        x_data = np.array(df[xlabel])
        xrange = (np.min(x) - 0.01, np.max(x) + 0.01)
    elif xrange[0] == None:
        x_data    = np.array(df[xlabel])
        xrange[0] = np.min(x) - 0.01
    elif xrange[1] == None:
        x_data    = np.array(df[xlabel])
        xrange[1] = np.max(x) + 0.01

    xline = np.linspace(xrange[0], xrange[1], nbins, endpoint=True)

    if verbose:
        iterations_list = tqdm(range(len(xline)))
    else:
        iterations_list = range(len(xline))

    if kernel_width is None:
        kernel_width = self.kernel_width

    #If kernel_width is just one number, then turn it
    #into an array of length = nbins
    if isinstance(kernel_width, (int, float)):
        kernel_width_array = np.ones(nbins)*kernel_width

    elif len(kernel_width) != len(xline):
        #Check if kernel_width has right dimensions
        raise ValueError("Size mismatch: kernel_width has length %d. Paramater nbins = %d"%(len(kernel_width), nbins))
    #Otherwise rename value for rest of code
    else:
        kernel_width_array = kernel_width

    x_data, y_data, z_data = np.array(df[xlabel]), np.array(df[ylabel]), np.array(df[zlabel])
    mask = np.invert(np.isinf(x_data)) & (np.invert(np.isinf(y_data))) & (np.invert(np.isinf(z_data)))
    x_data, y_data, z_data = x_data[mask], y_data[mask], z_data[mask]

    cov_corr = np.zeros([nBootstrap, len(xline)])

    for k in iterations_list:

        #Add mask to reduce data and improve
        #efficiency (hopefully)
        x_Mask = (x_data > xline[k] - 3*kernel_width_array[k]) & (x_data < xline[k] + 3*kernel_width_array[k])

        x_masked, y_masked, z_masked = x_data[x_Mask], y_data[x_Mask], z_data[x_Mask]

        if Output_mode.lower() in ['covariance', 'cov']:
            cov_corr[:, k] = lm.calc_covariance_fixed_x(x_masked, y_masked, z_masked, xline[k],
                                                        kernel_width=kernel_width_array[k],
                                                        parametric_bootstrap = True,
                                                        nBootstrap = nBootstrap)
        elif Output_mode.lower() in ['correlation', 'corr']:
            cov_corr[:, k] = lm.calc_correlation_fixed_x(x_masked, y_masked, z_masked, xline[k],
                                                         kernel_width=kernel_width_array[k],
                                                         parametric_bootstrap = True,
                                                         nBootstrap = nBootstrap)

    if xlog: xline = 10**xline

    if Output_mode.lower() in ['covariance', 'cov']:
        name = 'cov'
    elif Output_mode.lower() in ['correlation', 'corr']:
        name = 'corr'


    output_Data['x'] = xline

    output_Data['%s_%s_%s'%(name, ylabel, zlabel)]    = np.median(cov_corr, axis=0)
    output_Data['%s_%s_%s-'%(name, ylabel, zlabel)]   = np.percentile(cov_corr, 50 - 68/2, axis=0)
    output_Data['%s_%s_%s+'%(name, ylabel, zlabel)]   = np.percentile(cov_corr, 50 + 68/2, axis=0)
    output_Data['%s_%s_%s--'%(name, ylabel, zlabel)]  = np.percentile(cov_corr, 50 - 95/2, axis=0)
    output_Data['%s_%s_%s++'%(name, ylabel, zlabel)]  = np.percentile(cov_corr, 50 + 95/2, axis=0)
    output_Data['%s_%s_%s---'%(name, ylabel, zlabel)] = np.percentile(cov_corr, 50 - 99.7/2, axis=0)
    output_Data['%s_%s_%s+++'%(name, ylabel, zlabel)] = np.percentile(cov_corr, 50 + 99.7/2, axis=0)

    return output_Data
