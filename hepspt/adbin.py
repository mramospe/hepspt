'''
Module to manage special histograms, like adaptive binned 1D and 2D histograms.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Python
import numpy as np
import bisect, itertools


class AdBin:
    '''
    Represent a n-dimensional adaptive bin. This class is meant so serve
    as interface between the user and matplotlib to plot adaptive
    binned histograms.
    '''
    def __init__( self, arr, rg = None, wgts = None ):
        '''
        :param arr: array of data.
        :type arr: numpy.ndarray
        :param rg: range of the histogram (same length as "arr").
        :type rg: numpy.ndarray(float, float) or None.
        :param wgts: possible weights.
        :type wgts: numpy.ndarray or None
        '''
        arr, (vmin, vmax), wgts = _proc_hist_input(arr, rg, wgts)

        self.arr  = arr
        self.wgts = wgts
        self.vmin = vmin
        self.vmax = vmax

    def dens( self, arr, wgts = None ):
        '''
        :param arr: array of data to process.
        :type arr: numpy.ndarray
        :param wgts: possible weights.
        :type wgts: numpy.ndarray or None
        :returns: density of this bin.
        :rtype: float
        '''
        return self.sw(arr, wgts)/float(self.size())

    def size( self ):
        '''
        :returns: size of this bin calculated as the product of \
        the individual sizes in each dimension.
        :rtype: float
        '''
        return float(np.prod(self.vmax - self.vmin))

    def sw( self, arr, wgts = None ):
        '''
        :param arr: array of data to process.
        :type arr: numpy.ndarray
        :param wgts: possible weights.
        :type wgts: numpy.ndarray or None
        :returns: sum of weights for this bin.
        :rtype: float
        '''
        true = np.logical_and(arr >= self.vmin, arr < self.vmax).all(axis = 1)

        if wgts is not None:
            sw = wgts[true].sum()
        else:
            sw = np.count_nonzero(true)

        return float(sw)

    def divide( self, ndiv = 2 ):
        '''
        Divide this bin in two, using the median in each dimension. The
        dimension used to make the division is taken as that which generates
        the smallest bin.

        :param ndiv: number of divisions to create. For large values, this \
        algorithm will ask for having a low sum of weights for the first \
        bin, which will translate in having a long thin bin.
        :type ndiv: int
        :returns: two new bins, supposed to contain half the sum of weights of \
        the parent.
        :rtype: AdBin, AdBin
        '''
        assert ndiv > 1

        srt   = self.arr.argsort(axis = 0)
        sarr  = np.array([self.arr.T[i][s] for i, s in enumerate(srt.T)]).T
        swgts = self.wgts[srt]
        csw   = swgts.cumsum(axis = 0)

        co = np.max(csw)/float(ndiv)

        p = np.array([bisect.bisect_left(c, co) for c in csw.T])

        bounds = np.array([np.nextafter(sarr[p,i], np.infty)
                           for i, p in enumerate(p)])

        mask_left  = (self.arr < bounds)
        mask_right = (self.arr >= bounds)

        frags = np.array([min(self.arr[mask_left[:,i]].min(),
                              self.arr[mask_right[:,i]].min())
                          for i in xrange(self.arr.shape[1])])

        # The sample is cut following the criteria that leads to the
        # smallest bin possible.
        min_dim = frags.argmin()

        il = mask_left[:,min_dim]
        ir = mask_right[:,min_dim]

        left   = self.arr[il]
        right  = self.arr[ir]
        wleft  = self.wgts[il]
        wright = self.wgts[ir]

        lbd = np.array(self.vmax)
        lbd[min_dim] = bounds[min_dim]
        bl = AdBin(left, (self.vmin, lbd), wleft)

        rbd = np.array(self.vmin)
        rbd[min_dim] = bounds[min_dim]
        br = AdBin(right, (rbd, self.vmax), wright)

        # If the number of divisions is greater than 2, perform again the same
        # operation in the bin on the right
        all_bins = [bl]
        if ndiv > 2:
            all_bins += br.divide(ndiv - 1)
        else:
            all_bins.append(br)

        return all_bins


def adbin_hist1d( arr, nbins = 100, rg = None, wgts = None ):
    '''
    Create an adaptive binned histogram.

    :param arr: array of data.
    :type arr: numpy.ndarray
    :param nbins: number of bins.
    :type nbins: int
    :param rg: range of the histogram.
    :type rg: tuple(float, float) or None
    :param wgts: optional array of weights.
    :type wgts: numpy.ndarray or None
    :returns: values of the histogram and edges.
    :rtype: numpy.ndarray, numpy.ndarray
    '''
    arr, rg, wgts = _proc_hist_input_1d(arr, rg, wgts)

    # Sort the data
    srt  = arr.argsort()
    arr  = arr[srt]
    wgts = wgts[srt]

    # Solving the problem from the left and from the right reduces
    # the bias in the last edges
    le = adbin_hist1d_edges(arr, nbins, rg, wgts)
    re = adbin_hist1d_edges(arr[::-1], nbins, rg, wgts[::-1])[::-1]

    edges = (re + le)/2.

    vmin, vmax = rg
    edges[0]   = vmin
    edges[-1]  = vmax

    values, edges = np.histogram(arr, edges, weights = wgts)

    return values, edges


def adbin_hist1d_edges( arr, nbins = 100, rg = None, wgts = None ):
    '''
    Create adaptive binned edges from the given array.

    :param arr: array of data.
    :type arr: numpy.ndarray
    :param nbins: number of bins.
    :type nbins: int
    :param rg: range of the histogram.
    :type rg: tuple(float, float) or None
    :param wgts: optional array of weights.
    :type wgts: numpy.ndarray or None
    :returns: edges of the histogram, with size (nbins + 1).
    :rtype: numpy.ndarray
    '''
    arr, (vmin, vmax), wgts = _proc_hist_input_1d(arr, rg, wgts)

    edges = np.zeros(nbins + 1)

    for i in xrange(nbins - 1):

        csum = wgts.cumsum()
        reqs = csum[-1]/float(nbins - i)

        p = bisect.bisect_left(csum, reqs)
        if p == len(csum):
            p = -1

        s = csum[p]

        if s != reqs:
            # If the sum differs, then decide whether to use the
            # current index based on the relative difference.
            if p != 0 and np.random.uniform() < (s - reqs)/reqs:
                p -= 1

        edges[i + 1]  = (arr[p] + arr[p + 1])/2.

        arr  = arr[p + 1:]
        wgts = wgts[p + 1:]

    edges[0]  = vmin
    edges[-1] = vmax

    return edges


def adbin_hist2d( x, y, *args, **kwargs ):
    '''
    Create a 2D adaptive binned histogram. This function calls
    the adbin_histnd function.

    :param arr: array of data with the variables as columns.
    :type arr: numpy.ndarray
    :param nbins: number of bins.
    :type nbins: int
    :param wgts: optional array of weights.
    :type wgts: numpy.ndarray
    :returns: adaptive bins of the histogram, with size (nbins + 1).
    :rtype: list(AdBin)

    .. seealso: :func:`adbin_hist2d`
    '''
    return adbin_histnd(np.array([x, y]).T, *args, **kwargs)


def adbin_histnd( arr, nbins = 100, rg = None, wgts = None, ndiv = 2 ):
    '''
    Create a ND adaptive binned histogram.

    :param arr: array of data with the variables as columns.
    :type arr: numpy.ndarray
    :param rg: range of the histogram (same length as "arr").
    :type rg: numpy.ndarray(float, float) or None.
    :param nbins: number of bins. In this algorithm, divisions will be made \
    till the real number of bins is equal or greater than "nbins". If this \
    number is a power of "ndiv", then the real number of bins will match \
    "nbins".
    :type nbins: int
    :param wgts: optional array of weights.
    :type wgts: numpy.ndarray or None
    :param ndiv: see :meth:`AdBin.divide`.
    :type ndiv: int
    :returns: adaptive bins of the histogram, with size (nbins + 1).
    :rtype: list(AdBin)

    .. seealso: :meth:`AdBin.divide`
    '''
    assert len(arr) / nbins > 0

    bins = [AdBin(arr, rg, wgts)]
    while len(bins) < nbins:
        bins = list(itertools.chain.from_iterable(a.divide(ndiv) for a in bins))

    return bins


def _proc_hist_input_1d( arr, rg = None, wgts = None ):
    '''
    Process some of the input arguments of the functions to
    manage 1D histograms.

    :param arr: array of data.
    :type arr: numpy.ndarray
    :param rg: range of the histogram.
    :type rg: tuple(float, float) or None
    :param wgts: optional array of weights.
    :type wgts: numpy.ndarray or None
    :returns: processed array of data, weights, and the minimum \
    and maximum values.
    :rtype: numpy.ndarray, tuple(float, float), numpy.ndarray
    '''
    vmin, vmax = proc_range(arr, rg)

    cond = np.logical_and(arr >= vmin, arr < vmax)

    arr = arr[cond]

    if wgts is not None:
        wgts = wgts[cond]
    else:
        wgts = np.ones(len(arr))

    return arr, (vmin, vmax), wgts


def _proc_hist_input( arr, rg = None, wgts = None ):
    '''
    Process some of the input arguments of the functions to
    manage ND histograms.

    :param arr: array of data.
    :type arr: numpy.ndarray
    :param rg: range of the histogram in each dimension.
    :type rg: tuple(np.ndarray, np.ndarray) or None
    :param wgts: optional array of weights.
    :type wgts: numpy.ndarray or None
    :returns: processed array of data, weights, and the minimum \
    and maximum values for each dimension.
    :rtype: numpy.ndarray, tuple(np.ndarray, np.ndarray), numpy.ndarray
    '''
    vmin, vmax = proc_range(arr, rg)

    cond = np.logical_and(arr >= vmin, arr < vmax).all(axis = 1)

    arr = arr[cond]

    if wgts is not None:
        wgts = wgts[cond]
    else:
        wgts = np.ones(len(arr))

    return arr, (vmin, vmax), wgts


def proc_range( arr, rg = None ):
    '''
    Process the given range, determining the minimum and maximum
    values for a 1D histogram.

    :param arr: array of data.
    :type arr: numpy.ndarray
    :param rg: range of the histogram.
    :type rg: tuple(float, float) or None
    :returns: minimum and maximum values.
    :rtype: float, float
    '''
    if rg is not None:
        vmin, vmax = rg
    else:
        amax = arr.max(axis = 0)
        vmin = arr.min(axis = 0)
        vmax = np.nextafter(amax, 2*amax)

    return vmin, vmax
