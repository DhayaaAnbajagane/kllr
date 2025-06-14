"""
Kernel Localized Linear Regression (KLLR) method.

Introduction:
-------------

Linear regression of the simple least-squares variety has been a canonical method used to characterize
the relation between two variables, but its utility is limited by the fact that it reduces full
population statistics down to three numbers: a slope, normalization and variance/standard deviation.
With large empirical or simulated samples we can perform a more sensitive analysis
using a localized linear regression method (see, Farahi et al. 2018 and Anbajagane et al. 2020).
The KLLR method generates estimates of conditional statistics in terms of the local the slope, normalization,
and covariance. Such a method provides a more nuanced description of population statistics appropriate
for the very large samples with non-linear trends.

This code is an implementation of the Kernel Localized Linear Regression (KLLR) method
that performs a localized Linear regression described in Farahi et al. (2018). It employs
bootstrap re-sampling technique to estimate the uncertainties. We also provide a set of visualization
tools so practitioners can seamlessly generate visualization of the model parameters.


Quickstart:
-----------
To start using KLLR, simply use "from KLLR import kllr_model" to
access the primary functions and class. The exact requirements for the inputs are
listed in the docstring of the kllr_model() class further below.
An example for using KLLR looks like this:

    ------------------------------------------------------------------------
    |                                                                      |
    |    from kllr import kllr_model                                       |
    |                                                                      |
    |    lm = kllr_model(kernel_type = 'gaussian', kernel_width = 0.2)     |
    |    xrange, yrange_mean, intercept, slope, scatter =                  |
    |             lm.fit(x, y, nbins=11)                                   |
    |                                                                      |
    ------------------------------------------------------------------------

"""

import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn import linear_model, mixture
from sklearn.ensemble import RandomForestRegressor
import shap
import xgboost as xgb


def scatter_cal(x, y, slope, intercept, dof=None, weight=None):
    """
    This function computes the scatter about the mean relation.

    Parameters
    ----------
    x : numpy array
        Independent variable data vector.

    y : numpy array
        Dependent variable data vector.

    slope : float
        Slope of the regression model.

    intercept : float
        Intercept of the regression model.

    dof : int, optional
        Degree of freedom if known otherwise dof = len(x)

    weight : numpy array, optional
        Individual weights for each sample. If None it assume a uniform weight.


    Returns
    -------
    float
        The standard deviation of residuals about the mean relation

    """

    if len(x.shape) > 1 or len(y.shape) > 1:
        raise ValueError(
            "Incompatible dimension for X and Y. X and Y should be one dimensional numpy array,"
            ": len(X.shape) = %i while len(Y.shape) = %i." % (len(x.shape), len(y.shape)))

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "Incompatible dimension for X and Y. X and Y should have the same feature dimension,"
            ": X.shape[1] = %i while Y.shape[1] = %i." % (x.shape[0], y.shape[0]))

    if dof is None:
        dof = len(x)

    if weight is None:
        sig2 = sum((np.array(y) - (slope * np.array(x) + intercept)) ** 2) / dof
    else:
        sig2 = np.average((np.array(y) - (slope * np.array(x) + intercept)) ** 2, weights=weight)
        sig2 /= 1 - np.sum(weight**2)/np.sum(weight)**2

    return np.sqrt(sig2)

def multivariate_scatter_cal(X, y, slopes, intercept, dof=None, weight=None):
    """
    This function computes the scatter about the mean relation.

    Parameters
    ----------
    x : numpy array
        Independent variable data vector.

    y : numpy array
        Dependent variable data vector.

    slope : float
        Slope of the regression model.

    intercept : float
        Intercept of the regression model.

    dof : int, optional
        Degree of freedom if known otherwise dof = len(x)

    weight : numpy array, optional
        Individual weights for each sample. If None it assume a uniform weight.


    Returns
    -------
    float
        The standard deviation of residuals about the mean relation

    """

    if dof is None:
        dof = len(X)

    if weight is None:
        sig2 = sum((np.array(y) - (np.dot(X, slopes) + intercept)) ** 2) / dof
    else:
        sig2 = np.average((np.array(y) - (np.dot(X, slopes) + intercept)) ** 2, weights = weight)

    return np.sqrt(sig2)

def skewness_cal(x, y, slope, intercept, dof=None, weight=None):
    """
    This function computes the skewness about the mean relation.

    Parameters
    ----------
    x : numpy array
        Independent variable data vector.

    y : numpy array
        Dependent variable data vector.

    slope : float
        Slope of the regression model.

    intercept : float
        Intercept of the regression model.

    dof : int, optional
        Degree of freedom if known otherwise dof = len(x)

    weight : numpy array, optional
        Individual weights for each sample. If None it assume a uniform weight.


    Returns
    -------
    float
        The skewness of residuals about the mean relation

    """

    if len(x.shape) > 1 or len(y.shape) > 1:
        raise ValueError(
            "Incompatible dimension for X and Y. X and Y should be one dimensional numpy array,"
            ": len(X.shape) = %i while len(Y.shape) = %i." % (len(x.shape), len(y.shape)))

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "Incompatible dimension for X and Y. X and Y should have the same feature dimension,"
            ": X.shape[0] = %i while Y.shape[0] = %i." % (x.shape[0], y.shape[0]))

    if dof is None:
        dof = len(x)

    if weight is None:
        m2 = sum((np.array(y) - (slope * np.array(x) + intercept)) ** 2) / dof
        m3 = sum((np.array(y) - (slope * np.array(x) + intercept)) ** 3) / dof

    else:
        m2 = np.average((np.array(y) - (slope * np.array(x) + intercept)) ** 2, weights=weight)
        m3 = np.average((np.array(y) - (slope * np.array(x) + intercept)) ** 3, weights=weight)

    skew = m3/m2**(3/2)

    return skew

def multivariate_skewness_cal(X, y, slopes, intercept, dof=None, weight=None):
    """
    This function computes the skewness about the mean relation, but for
    multivariate linear regression.

    Parameters
    ----------
    X : numpy array
        Independent variable data vector. Can have multiple features

    y : numpy array
        Dependent variable data vector.

    slope : numpy array
        1D array of the slopes of the regression model.
        Each entry is the slope of a particular feature.

    intercept : float
        Intercept of the regression model.

    dof : int, optional
        Degree of freedom if known otherwise dof = len(x)

    weight : numpy array, optional
        Individual weights for each sample. If None it assume a uniform weight.


    Returns
    -------
    float
        The standard deviation of residuals about the mean relation

    """

    if len(X.shape) != 2:
        raise ValueError(
            "Incompatible dimension for X. X should be a one dimensional numpy array,"
            ": len(X.shape) = %i." %len(X.shape))

    if len(y.shape) != 1:
        raise ValueError(
            "Incompatible dimension for Y. Y should be a one dimensional numpy array,"
            ": len(Y.shape) = %i." %len(Y.shape))

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "Incompatible dimension for X and Y. X and Y should have the same feature dimension,"
            ": X.shape[0] = %i while Y.shape[0] = %i." % (X.shape[0], y.shape[0]))

    if dof is None:
        dof = len(X)

    if weight is None:
        m2 = sum((np.array(y) - (np.dot(X, slopes) + intercept)) ** 2) / dof
        m3 = sum((np.array(y) - (np.dot(X, slopes) + intercept)) ** 3) / dof

    else:
        m2 = np.average((np.array(y) - (np.dot(X, slopes) + intercept)) ** 2, weights=weight)
        m3 = np.average((np.array(y) - (np.dot(X, slopes) + intercept)) ** 3, weights=weight)

    skew = m3/m2**(3/2)

    return skew

def calculate_weigth(x, kernel_type='gaussian', mu=0, width=0.2):
    """
    According to the provided kernel, this function computes the weight assigned to each data point.

    Parameters
    ----------
    x : numpy array
        A one dimensional data vector.

    kernel_type : string, optional
        The kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel. The default is Gaussian

    mu, width : float, optional
        If kernel_type = 'gaussian' then 'mu' and 'width' are the mean and width of the gaussian kernels, respectively.
        If kernel_type = 'uniform' then 'mu' and 'width' are the mean and width of the uniform kernels, respectively.

    Returns
    -------
    float
        the weight vector
    """

    if len(x.shape) > 1:
        raise ValueError(
            "Incompatible dimension for X. X  should be one dimensional numpy array,"
            ": len(X.shape) = %i." % (len(x.shape)))

    # the gaussian kernel
    def gaussian_kernel(x, mu=0.0, width=1.0):
        return np.exp(-(x - mu) ** 2 / 2. / width ** 2)

    # the uniform kernel
    def uniform_kernel(x, mu=0.0, width=1.0):
        w = np.zeros(len(x))
        w[(x - mu < width / 2.0) * (x - mu > -width / 2.0)] = 1.0
        return w

    if kernel_type == 'gaussian':
        w = gaussian_kernel(x, mu=mu, width=width)
    elif kernel_type == 'uniform':
        w = uniform_kernel(x, mu=mu, width=width)
    else:
        print("Warning : ", kernel_type, "is not a defined filter.")
        print("It assumes w = 1 for every point.")
        w = np.ones(len(x))

    return w

class kllr_model():
    """
    A class used to represent a KLLR model and perform the fit. It is supported bu additional functions that allows
     to compute the conditional properties such as residuals about the mean relation,
     the correlation coefficient, and the covariance.

    Attributes
    ----------
    kernel_type : string
        The kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel. The default is Gaussian

    kernel_width : float
         If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
         If kernel_type = 'uniform' then 'width' is the width of the uniform kernel.

    Methods
    -------
    linear_regression(x, y, weight = None)
        perform a linear regression give a set of weights

    subsample(x, length=False)
        generate a bootstrapped sample

    calc_correlation_fixed_x(self, data_x, data_y, data_z, x, kernel_type = None, kernel_width = None)
        compute the conditional correlation coefficient conditioned at point x

    calc_covariance_fixed_x(x, y, xrange = None, nbins = 60, kernel_type = None, kernel_width = None)
        compute the conditional correlation coefficient conditioned at point x

    calculate_residual(x, y, xrange = None, nbins = 60, kernel_type = None, kernel_width = None)
        compute residuls about the mean relation i.e., res = y - <y|X>

    PDF_generator(self, res, nbins, nBootstrap = 1000, funcs = {}, xrange = (-4, 4), verbose = True,  **kwargs)
        generate a binned PDF of residuasl around the mean relation

    fit(x, y, xrange = None, nbins = 25, kernel_type = None, kernel_width = None)
        fit a kernel localized linear relation to (x, y) pairs, i.e. <y | x> = a(y) x + b(y)

    """

    def __init__(self, kernel_type='gaussian', kernel_width=0.2):
        """
        Parameters
        ----------
        kernel_type : string, optional
            the kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel. The default is Gaussian

        kernel_width : float, optional
            if kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            if kernel_type = 'uniform' then 'width' is the width of the uniform kernel.
        """

        self.kernel_type = kernel_type
        self.kernel_width = kernel_width

    def linear_regression(self, x, y, weight=None):
        """
        This function perform a linear regression given a set of weights and return the normalization, slope, and
        scatter about the mean relation.

        Parameters
        ----------
        x : numpy array
            Independent variable data vector. This version only support a one dimensional data vector.

        y : numpy array
            Dependent variable data vector. This version only support a one dimensional data vector.

        weight : float, optional
            Individual weights for each sample. If none it assumes a uniform weight.

        Returns
        -------
        float
             intercept

        float
             slope

        float
             scatter about the mean relation
        """

        if weight is None:
            slope, intercept, r_value, p_value = stats.linregress(x, y)[0:4]
        else:
            regr = linear_model.LinearRegression()
            # Train the model using the training sets
            regr.fit(x[:, np.newaxis], y, sample_weight=weight)
            slope = regr.coef_[0]
            intercept = regr.intercept_

        sig  = scatter_cal(x,  y, slope, intercept, weight=weight)
        skew = skewness_cal(x, y, slope, intercept, weight=weight)

        return intercept, slope, sig, skew

    def multivariate_linear_regression(self, X, y, weight=None):
        """
        This function performs a multivariate linear regression given a set of weights and
        returns the normalization, slope, and scatter about the mean relation. The weights
        are applied to only the first feature, i.e. X[:, 0].

        Parameters
        ----------
        X : numpy array
            Independent variable data vector. Can have multiple features.

        y : numpy array
            Dependent variable data vector. This version only support a one dimensional data vector.

        weight : float, optional
            Individual weights for each sample. If none it assumes a uniform weight.

        Returns
        -------
        float
             intercept

        float
             slope

        float
             scatter about the mean relation
        """

        #if X is not 2D then raise error
        if len(X.shape) != 2:
            raise ValueError("Incompatible dimension for X."
                             "X should be two dimensional numpy array.")

        # Initialize regressor
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(X, y, sample_weight=weight)
        slopes = regr.coef_
        intercept = regr.intercept_

        sig  = multivariate_scatter_cal(X,  y, slopes, intercept, weight=weight)
        skew = multivariate_skewness_cal(X, y, slopes, intercept, weight=weight)

        return intercept, slopes, sig, skew

    def subsample(self, x, length=False):
        """
        This function re-samples an array and returns the re-sampled array
        and its indices (if you need to use it as a mask for other arrays)

        Parameters
        ----------
        x : numpy array
            One dimensional data array.

        length : bool, optional
            The length of bootstrapped sample. If False it assumes `length = len(x)`.

        Returns
        -------
        numpy array
            the re-sampled vector

        numpy array
            the re-sample index
        """
        x = np.array(x)
        l = length if length else len(x)
        resample = np.floor(np.random.rand(l) * int(len(x))).astype(int)
        return x[resample], resample

    def calc_correlation_fixed_x(self, data_x, data_y, data_z, x, kernel_type=None, kernel_width=None,
                                 parametric_bootstrap = False, nBootstrap = 1000):
        """
        This function computes the conditional correlation between two variables data_y and data_z at point x.

        Parameters
        ----------
        data_x : numpy array
            Independent variable data vector. This version only support a one dimensional data vector.

        data_y : numpy array
            Dependent variable data vector. This version only support a one dimensional data vector.

        data_z : numpy array
            Dependent variable data vector. This version only support a one dimensional data vector.

        x : float
            Value of the conditional parameter. It computes the correlation coefficient at this point.

        kernel_type : string, optional
            Rhe kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel.
            If None it uses the pre-specified `kernel_type`

        kernel_width : float, optional
            If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            If kernel_type = 'uniform' then 'width' is the width of the uniform kernel.
            If None it uses the pre-specified `kernel_width`

        Returns
        -------
        float
             Correlation coefficient.
        """

        if kernel_type is not None:
            self.kernel_type = kernel_type

        if kernel_width is not None:
            self.kernel_width = kernel_width

        if parametric_bootstrap:

            Output = np.empty(nBootstrap)

        weight = calculate_weigth(data_x, kernel_type=self.kernel_type, mu=x, width=self.kernel_width)

        intercept, slope, sig, skew = self.linear_regression(data_x, data_y, weight=weight)
        dy = data_y - slope * data_x - intercept

        intercept, slope, sig, skew = self.linear_regression(data_x, data_z, weight=weight)
        dz = data_z - slope * data_x - intercept

        if parametric_bootstrap:

            for i in range(nBootstrap):

                if i == 0:
                    index = np.arange(dy.size)
                else:
                    index = self.subsample(dy)[1]

                sig = np.cov(dy[index], dz[index], aweights=weight[index])

                Output[i] = sig[1, 0] / np.sqrt(sig[0, 0] * sig[1, 1])
        else:
            sig = np.cov(dy, dz, aweights=weight)
            Output = sig[1, 0] / np.sqrt(sig[0, 0] * sig[1, 1])

        return Output

    def calc_covariance_fixed_x(self, data_x, data_y, data_z, x, kernel_type=None, kernel_width=None,
                                parametric_bootstrap = False, nBootstrap = 1000):
        """
        This function computes the conditional covariance between two variables data_y and data_z at point x.

        Parameters
        ----------
        data_x : numpy array
            Independent variable data vector. This version only support a one dimensional data vector.

        data_y : numpy array
            Dependent variable data vector. This version only support a one dimensional data vector.

        data_z : numpy array
            Dependent variable data vector. This version only support a one dimensional data vector.

        x : float
            Value of the conditional parameter. It computes the covariance at this point.

        kernel_type : string, optional
            Rhe kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel.
            If None it uses the pre-specified `kernel_type`

        kernel_width : float, optional
            If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            If kernel_type = 'uniform' then 'width' is the width of the uniform kernel.
            If None it uses the pre-specified `kernel_width`

        Returns
        -------
        float
             Covariance.
        """

        if kernel_type is not None:
            self.kernel_type = kernel_type

        if kernel_width is not None:
            self.kernel_width = kernel_width

        if parametric_bootstrap:

            Output = np.empty(nBootstrap)

        weight = calculate_weigth(data_x, kernel_type=self.kernel_type, mu=x, width=self.kernel_width)

        intercept, slope, sig, skew = self.linear_regression(data_x, data_y, weight=weight)
        dy = data_y - slope * data_x - intercept

        intercept, slope, sig, skew = self.linear_regression(data_x, data_z, weight=weight)
        dz = data_z - slope * data_x - intercept

        if parametric_bootstrap:

            for i in range(nBootstrap):

                if i == 0:
                    index = np.arange(dy.size)
                else:
                    index = self.subsample(dy)[1]

                sig = np.cov(dy[index], dz[index], aweights=weight[index])

                Output[i] = sig[1, 0]
        else:
            sig = np.cov(dy, dz, aweights=weight)
            Output = sig[1, 0]

        return Output

    def calculate_residual(self, x, y, xrange=None, nbins=60, kernel_type=None, kernel_width=None):
        """
        This function computes the residuals about the mean relation, i.e. res = y - <y | x>.

        Parameters
        ----------
        x : numpy array
            Independent variable data vector. This version only support a one dimensional data vector.

        y : numpy array
            Dependent variable data vector. This version only support a one dimensional data vector.

        xrange : list, optional
            The range of regression. The first element is the min and the second element is the max.
            If None it set it to min and max of x, i.e., `xrange = [min(x), max(x)]`

        nbins : int, optional
            The numbers of bins to compute the local regression parameters. The default value is 60 bins.

        kernel_type : string, optional
            The kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel.
            If None it uses the pre-specified `kernel_type`

        kernel_width : float, optional
            If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            If kernel_type = 'uniform' then 'width' is the width of the uniform kernel.
            If None it uses the pre-specified `kernel_width`

        Returns
        -------
        numpy array
             Individual residuals.
        """

        # NOTE: The number of sampling points (30) is currently used as default option
        # changing it only changes accuracy a bit (narrower bins means interpolations are more accurate),
        # and also changes computation time
        if xrange == None:
            xrange = (np.min(x) - 0.01, np.max(x) + 0.01)
        elif xrange[0] == None:
            xrange = [np.min(x) - 0.01, xrange[1]]
        elif xrange[1] == None:
            xrange = [xrange[0], np.max(x) + 0.01]

        xline = np.linspace(xrange[0], xrange[1], nbins, endpoint=True)

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

        if kernel_type is not None:
            self.kernel_type = kernel_type


        res = np.array([])  # Array to store residuals
        Index = np.array([])  # array to map what residual belongs to what Halo

        # Loop over each bin defined by the bin edges above
        for i in range(len(xline) - 1):
            # Compute weight at each edge
            w1 = calculate_weigth(x, kernel_type=self.kernel_type, mu=xline[i], width=kernel_width_array[i])
            w2 = calculate_weigth(x, kernel_type=self.kernel_type, mu=xline[i + 1], width=kernel_width_array[i + 1])

            # Compute expected y-value at each bin-edge
            intercept1, slope1, scatter1, _ = self.linear_regression(x, y, weight=w1)
            yline1 = slope1 * xline[i] + intercept1
            intercept2, slope2, scatter2, _ = self.linear_regression(x, y, weight=w2)
            yline2 = slope2 * xline[i + 1] + intercept2

            # Compute slope in this bin
            slope = (yline2 - yline1) / (xline[i + 1] - xline[i])

            # Mask to select only halos in this bin
            mask = (x >= xline[i]) & (x < xline[i + 1])

            # Interpolate to get scatter at each halo
            std = scatter1 + (scatter2 - scatter1) / (xline[i + 1] - xline[i]) * (x[mask] - xline[i])
            # Interpolate expected y-values and Compute residuals
            dy = y[mask] - (yline1 + (yline2 - yline1) / (xline[i + 1] - xline[i]) * (x[mask] - xline[i]))
            res = np.concatenate((res, dy / std))

            # Keep track of an index that maps which residual belongs to which halo
            Index = np.concatenate((Index, np.where(mask)[0]))

        # Reshuffle residuals so that res[i] was computed using the halo with values x[i] and y[i]
        res = np.array(res)[np.argsort(Index)]

        return res

    def PDF_generator(self, res, nbins=20, nBootstrap=1000, funcs={}, xrange=(-4, 4), verbose=True,
                      density=True, weights=None):
        """

        Parameters
        ----------
        res : numpy array
            Individual residuals, i.e. res = y - <y|x>.

        nbins : integer, optional
            Number of bins for the PDF.

        xrange : list, optional
            Tuple containing min and max bin values.

        nBootstrap : integer, optional
            Number of Bootstrap realizations of the PDF.

        funcs : dictionary, optional
            Dictionary of functions to apply on the Bootstrapped residuals. Format is {'Name': func}.

        verbose : bool, optional
            Turn on/off the verbosity of the PDF output during the bootstrapping.

        density : bool, optional
            If False, the result will contain the number of samples in each bin.
             If True, the result is the value of the probability density function at the bin, normalized such
             that the integral over the range is 1. Note that the sum of the histogram values will not be
             equal to 1 unless bins of unity width are chosen; it is not a probability mass function.

        weights : numpy array, optional
            An array of weights, of the same shape as a. Each value in a only contributes its associated weight
             towards the bin count (instead of 1). If density is True, the weights are normalized, so that the
             integral of the density over the range remains 1. If None it assumes a uniform weight.

        Returns
        -------
        numpy array
            Numpy array of size (nBootstrap, nbins) containing all realizations of PDFs

        numpy array
            Central values of the bins of the PDF

        Dictionary
            Dictionary with format {'name': result}, where result is the output of user-inputted inputted
            functions acting on residuals.
        """

        if verbose:
            iterations = tqdm(range(nBootstrap))
        else:
            iterations = range(nBootstrap)

        Output = {}

        # Writing a flag for array creation in case user pases an array defining the bin_edges
        # instead of numbers of bins.
        if isinstance(nbins, np.ndarray):
            print("An Array has been passed into nbins param")
            PDFs = np.empty([nBootstrap, len(nbins) - 1])
        else:
            PDFs = np.empty([nBootstrap, nbins])

        # Generate dictionary whose keys are the same keys as the 'func' dictionary
        for function_name in funcs:
            # Choose to use list over np.array because list can handle many types of data
            Output[function_name] = []

        for iBoot in iterations:

            if iBoot == 0:
                residuals = res
                try:
                    w = weights
                except:
                    w = None
            else:
                residuals, index = self.subsample(res)
                try:
                    # If weights exist, reshuffle them according to subsampling
                    w = weights[index]
                except:
                    w = None

            # Compute PDF and store in one row of 2D array
            PDFs[iBoot, :] = np.histogram(residuals, bins=nbins, range=xrange, weights=w, density=density)[0]

            # For each bootstrap, store function output for each function in funcs
            for function_name in funcs:
                Output[function_name].append(funcs[function_name](residuals))

        if isinstance(nbins, np.ndarray):
            # Set array to be the centers of each bin defined by the input array, nbin
            bins = (nbins[1:] + nbins[:-1]) / 2.
        elif isinstance(nbins, (int)):
            # Generate bin_edges
            bins = np.histogram([], bins=nbins, range=xrange)[1]
            # Set array to be the centers of each bin
            bins = (bins[1:] + bins[:-1]) / 2.

        return PDFs, bins, Output

    def fit(self, x, y, xrange=None, nbins=25, kernel_type=None, kernel_width=None):
        """
        This function computes the local regression parameters at the points within xrange.

        Parameters
        ----------
        x : numpy array
            Independent variable data vector. This version only support a one dimensional data vector.

        y : numpy array
            Dependent variable data vector. This version only support a one dimensional data vector.

        xrange : list, optional
            The first element is the min and the second element is the max,
            If None, it sets xrange to [min(x), max(x)]

        nbins : int, optional
            The numbers of data points to compute the local regression parameters

        kernel_type : string, optional
            The kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel.
            If None it uses the pre-specified `kernel_type`

        kernel_width : float, optional
            If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            If kernel_type = 'uniform' then 'width' is the width of the uniform kernel.
            If None it uses the pre-specified `kernel_width`.

        Returns
        -------
        numpy-array
            The local points.

        numpy-array
            The mean value at the local points

        numpy-array
            The intercept at the local points

        numpy-array
            The slope at the local points

        numpy-array
            The scatter around mean relation
        """

        # Define x_values to compute regression parameters at
        if xrange == None:
            xrange = (np.min(x) - 0.01, np.max(x) + 0.01)
        elif xrange[0] == None:
            xrange[0] = np.min(x) - 0.01
        elif xrange[1] == None:
            xrange[1] = np.max(x) + 0.01
        xline = np.linspace(xrange[0], xrange[1], nbins, endpoint=True)

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


        if kernel_type is not None:
            self.kernel_type = kernel_type

        # Generate array to store output from fit
        # Generate array to store output from fit
        yline_exp, slope_exp, intercept_exp, scatter_exp, skew_exp = [np.zeros(xline.size)
                                                                      for i in range(5)]

        # loop over every sample point
        for i in range(len(xline)):
            # Generate weights at that sample point
            w = calculate_weigth(x, kernel_type=self.kernel_type, mu=xline[i], width=kernel_width_array[i])
            # Compute fit params using linear regressions
            intercept_exp[i], slope_exp[i], scatter_exp[i], skew_exp[i] = self.linear_regression(x, y, weight=w)
            # Generate expected y_value using fit params
            yline_exp[i] = slope_exp[i] * xline[i] + intercept_exp[i]

        return xline, yline_exp, intercept_exp, slope_exp, scatter_exp, skew_exp


    def fit_bootstrapped(self, x, y, xrange=None, nbins=25, kernel_type=None, kernel_width=None, nBootstrap = 1):
        """
        This function computes the local regression parameters at the points within xrange.

        Parameters
        ----------
        x : numpy array
            Independent variable data vector. This version only support a one dimensional data vector.

        y : numpy array
            Dependent variable data vector. This version only support a one dimensional data vector.

        xrange : list, optional
            The first element is the min and the second element is the max,
            If None, it sets xrange to [min(x), max(x)]

        nbins : int, optional
            The numbers of data points to compute the local regression parameters

        kernel_type : string, optional
            The kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel.
            If None it uses the pre-specified `kernel_type`

        kernel_width : float, optional
            If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            If kernel_type = 'uniform' then 'width' is the width of the uniform kernel.
            If None it uses the pre-specified `kernel_width`.

        Returns
        -------
        numpy-array
            The local points.

        numpy-array
            The mean value at the local points

        numpy-array
            The intercept at the local points

        numpy-array
            The slope at the local points

        numpy-array
            The scatter around mean relation
        """

        # Define x_values to compute regression parameters at
        if xrange == None:
            xrange = (np.min(x) - 0.01, np.max(x) + 0.01)
        elif xrange[0] == None:
            xrange[0] = np.min(x) - 0.01
        elif xrange[1] == None:
            xrange[1] = np.max(x) + 0.01
        xline = np.linspace(xrange[0], xrange[1], nbins, endpoint=True)

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


        if kernel_type is not None:
            self.kernel_type = kernel_type

        # Generate array to store output from fit
        # Generate array to store output from fit
        yline_exp, slope_exp, intercept_exp, scatter_exp, skew_exp = [np.zeros([nBootstrap, xline.size])
                                                                      for i in range(5)]

        # loop over every sample point
        for i in tqdm(range(len(xline))):

            x_Mask = (x > xline[i] - kernel_width_array[i]*3) & (x < xline[i] + kernel_width_array[i]*3)

            x_smaller = x[x_Mask]
            y_smaller = y[x_Mask]
            # Generate weights at that sample point
            w = calculate_weigth(x_smaller, kernel_type=self.kernel_type, mu=xline[i], width=kernel_width_array[i])

            for nBoot in range(nBootstrap):

                subsample_index = np.random.randint(0, x_smaller.size, x_smaller.size)
                # Compute fit params using linear regressions
                intercept_exp[nBoot, i], slope_exp[nBoot, i], scatter_exp[nBoot, i], skew_exp[nBoot, i] = self.linear_regression(x_smaller[subsample_index], y_smaller[subsample_index], weight=w[subsample_index])
                # Generate expected y_value using fit params
                yline_exp[nBoot, i] = slope_exp[nBoot, i] * xline[i] + intercept_exp[nBoot, i]

        if nBootstrap == 1:

            yline_exp = np.squeeze(yline_exp, 0)
            intercept_exp = np.squeeze(intercept_exp, 0)
            slope_exp = np.squeeze(slope_exp, 0)
            scatter_exp = np.squeeze(scatter_exp, 0)
            skew_exp = np.squeeze(skew_exp, 0)

        return xline, yline_exp, intercept_exp, slope_exp, scatter_exp, skew_exp


    def multivariate_fit(self, X, y, xrange=None, nbins=25, kernel_type=None, kernel_width=None):
        """
        This function computes the local regression parameters at the points within xrange.
        This version support supports multidimensional data for the independent data matrix, X.
        However the kernel weighting uses only the first column in X, i.e. X[:, 0].
        The predicted variable, y, must still be 1D.

        Parameters
        ----------
        X : numpy array (n_rows, n_features)
            Independent variable data matrix.
            The weighting is only applied to the first feature, however.

        y : numpy array (n_rows)
            Dependent variable data vector. Must be a one dimensional data vector.

        xrange : list, optional
            The first element is the min and the second element is the max,
            If None, it sets xrange to [min(x), max(x)]

        nbins : int, optional
            The numbers of data points to compute the local regression parameters

        kernel_type : string, optional
            The kernel type, ['gaussian', 'uniform'] else it assumes uniform kernel.
            If None it uses the pre-specified `kernel_type`

        kernel_width : float, optional
            If kernel_type = 'gaussian' then 'width' is the width of the gaussian kernel.
            If kernel_type = 'uniform' then 'width' is the width of the uniform kernel.
            If None it uses the pre-specified `kernel_width`.

        Returns
        -------
        numpy-array
            The local points.

        numpy-array
            The mean value at the local points

        numpy-array
            The intercept at the local points

        numpy-array
            The slope at the local points

        numpy-array
            The scatter around mean relation
        """

        # Define x_values to compute regression parameters at
        if xrange == None:
            xrange = (np.min(X[:, 0]), np.max(X[:, 0]))
        elif xrange[0] == None:
            xrange[0] = np.min(X[:, 0])
        elif xrange[1] == None:
            xrange[1] = np.max(X[:, 0])
        xline = np.linspace(xrange[0], xrange[1], nbins, endpoint=True)

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


        if kernel_type is not None:
            self.kernel_type = kernel_type

        # Generate array to store output from fit
        intercept_exp, slope_exp, scatter_exp, skew_exp = [np.zeros(xline.size) for i in range(4)]

        slope_exp = np.zeros([xline.size, X.shape[1]])
        # loop over every sample point
        for i in range(xline.size):

            # Generate weights at that sample point
            w = calculate_weigth(X[:, 0], kernel_type=self.kernel_type, mu=xline[i], width=kernel_width_array[i])

            # Compute fit params using multivariate linear regression
            intercept_exp[i], slope_exp[i, :], scatter_exp[i], skew_exp[i] = self.multivariate_linear_regression(X, y, weight=w)

            # Doesn't seem clear on what it means to
            # compute mean relation just as a function
            # of M200c even though we regress on all properties
            # so I don't give expected <y | x_0> here (eg. x_0 --> M200c)

        return xline, slope_exp, scatter_exp, skew_exp

    def Shapley_values(self, x, y, Z, xrange = None, nbins = 25, kernel_type=None, kernel_width=None,
                       n_jobs = 1, n_estimators = 100, max_samples = None):
        """
        Given an input matrix Z and target array y, provide a weighted estimate
        of how much each feature in Z contributes to the values in y. Weighting is
        done using array 'x'.
        """

        # Define x_values to compute regression parameters at
        if xrange == None:
            xrange = (np.min(x) - 0.01, np.max(x) + 0.01)
        elif xrange[0] == None:
            xrange[0] = np.min(x) - 0.01
        elif xrange[1] == None:
            xrange[1] = np.max(x) + 0.01
        xline = np.linspace(xrange[0], xrange[1], nbins, endpoint=True)

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

        if kernel_type is not None:
            self.kernel_type = kernel_type

        #Output array
        Nmax     = 2000
        phi_SHAP = np.ones(shape = (nbins, Nmax, Z.shape[1]))*np.nan


        #Loop over sample points
        for i in tqdm(range(xline.size)):

            #Select only data within 1sigma of mass scale
            Mask = (x < xline[i] + kernel_width_array[i]) & (x > xline[i] - kernel_width_array[i])
            # Mask = (x < xline[i] + 0.1) & (x > xline[i] - 0.1)

            #Setup RF Regressor and find most optimal parameters
            params = {}

            model  = xgb.train(params, xgb.DMatrix(Z[Mask, :], y[Mask]), num_boost_round=100)

            #Extract SHAP value
            N_limit = 2000
            if N_limit < Mask.sum():
                subsample_index = np.random.randint(0, Mask.sum(), N_limit)
                index = np.where(Mask)[0][subsample_index]
                max   = Nmax
            else:
                index = np.where(Mask)[0]
                max   = np.sum(Mask)

            phi_SHAP[i, :max, :] = (shap.TreeExplainer(model, data = None)
                                         .shap_values(Z[index, :], check_additivity = True))

            # import matplotlib.pyplot as plt
            # plt.figure(figsize = (12,8))
            # plt.grid()
            # for j in range(Z.shape[1]):
            #     No_nans = phi_SHAP[i, index, j] > -1000000000
            #     plt.hist(phi_SHAP[i, index, j][No_nans], bins = 30, alpha = 0.3)
            # plt.show()
        return xline, phi_SHAP
