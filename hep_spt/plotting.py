'''
Provide some useful functions to plot with matplotlib.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Local
from hep_spt import __project_path__
from hep_spt.math_aux import lcm
from hep_spt.stats import poisson_freq_uncert_one_sigma

# Python
import matplotlib.pyplot as plt
import numpy as np
import os
from cycler import cycler


__all__ = ['PlotVar', 'errorbar_hist', 'process_range', 'samples_cycler', 'set_style', 'text_in_rectangles']


class PlotVar:
    '''
    Define an object to store plotting information, like a name, a title,
    a binning scheme and a range.
    '''
    def __init__( self, name, title = None, bins = 20, rg = None ):
        '''
        :param name: name of the variable.
        :type name: str
        :param title: title of the variable.
        :type title: str
        :param bins: see the argument "bins" in :func:`numpy.histogram`.
        :type bins: int or sequence of scalars or str
        :param rg: range of this variable (min, max).
        :type rg: tuple(float, float)
        '''
        self.name  = name
        self.title = title or name
        self.bins  = bins
        self.rg    = rg


def errorbar_hist( arr, bins = 20, rg = None, wgts = None, norm = False ):
    '''
    Calculate the values needed to create an error bar histogram.

    :param arr: input array of data to process.
    :param bins: see :func:`numpy.histogram`.
    :type bins: int or sequence of scalars or str
    :param rg: range to process in the input array.
    :type rg: tuple(float, float)
    :param wgts: possible weights for the histogram.
    :type wgts: collection(value-type)
    :returns: values, edges, the spacing between bins in X the Y errors. \
    In the non-weighted case, errors in Y are returned as two arrays, with the \
    lower and upper uncertainties.
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
    '''
    if wgts is not None:
        # Use sum of the square of weights to calculate the error
        sw2, edges = np.histogram(arr, bins, rg, weights = wgts)

        values, _ = np.histogram(arr, edges, weights = wgts)

        ey = np.sqrt(sw2)

    else:
        # Use the poissonian errors
        values, edges = np.histogram(arr, bins, rg, weights = wgts)

        # For compatibility with matplotlib.pyplot.errorbar
        ey = poisson_freq_uncert_one_sigma(values).T

    ex = (edges[1:] - edges[:-1])/2.

    if norm:

        s = float(values.sum())

        if s != 0:
            values /= s
            ey /= s
        else:
            ey *= np.finfo(ey.dtype).max

    return values, edges, ex, ey


def process_range( arr, rg = None ):
    '''
    Process the given range, determining the minimum and maximum
    values for a 1D histogram.

    :param arr: array of data.
    :type arr: numpy.ndarray
    :param rg: range of the histogram. It must contain tuple(min, max), \
    where "min" and "max" can be either floats (1D case) or collections \
    (ND case).
    :type rg: tuple or None
    :returns: minimum and maximum values.
    :rtype: float, float
    '''
    if rg is not None:
        vmin, vmax = rg
    else:
        amax = arr.max(axis = 0)
        vmin = arr.min(axis = 0)
        vmax = np.nextafter(amax, np.infty)

    return vmin, vmax


def samples_cycler( smps, *args, **kwargs ):
    '''
    Often, one wants to plot several samples with different matplotlib
    styles. This function allows to create a cycler.cycler object
    to loop over the given samples, where the "label" key is filled
    with the values from "smps".

    :param smps: list of names for the samples.
    :type smps: collection(str)
    :param args: position argument to cycler.cycler.
    :type args: tuple
    :param kwargs: keyword arguments to cycler.cycler.
    :type kwargs: dict
    :returns: cycler object with the styles for each sample.
    :rtype: cycler.cycler
    '''
    cyc = cycler(*args, **kwargs)

    ns = len(smps)
    nc = len(cyc)

    if ns > nc:

        warnings.warn('Not enough plotting styles in cycler, '\
                      'some samples might have the same style.')

        l = math_aux.lcm(ns, nc)

        re_cyc = (l*cyc)[:ns]
    else:
        re_cyc = cyc[:ns]

    return re_cyc + cycler(label = smps)


def set_style():
    '''
    Set the default style for matplotlib to that from this project.
    '''
    plt.style.use(os.path.join(__project_path__, 'mpl/hep_spt.mplstyle'))


def text_in_rectangles( ax, recs, txt, **kwargs ):
    '''
    Write text inside matplotlib.patches.Rectangle instances.

    :param ax: axes where the rectangles are being drawn.
    :type ax: matplotlib.axes.Axes
    :param recs: set of rectangles to work with.
    :type recs: collection(matplotlib.patches.Rectangle)
    :param txt: text to fill with in each rectangle.
    :type txt: collection(str)
    :param kwargs: any other argument to matplotlib.axes.Axes.annotate.
    :type kwargs: dict
    '''
    for r, t in zip(recs, txt):
        x, y = r.get_xy()

        cx = x + r.get_width()/2.
        cy = y + r.get_height()/2.

        ax.annotate(t, (cx, cy), **kwargs)
