'''
Examples related to the "adbin" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Custom
import hepspt

# Python
import matplotlib.pyplot as plt
import numpy as np


def main():
    '''
    Main function called when executing this module as a script.
    '''
    size = 1000
    bins = 20
    rg   = (-10, 10)

    sample  = np.random.normal(0., 2, size)
    weights = np.random.uniform(0, 1, size)

    figs, (root, al, ar) = plt.subplots(1, 3)

    root.hist(sample, bins, rg, label = 'raw')
    root.set_title('raw')
    root.set_ylabel('entries')

    for s, w, a, t in ((sample, None, al, 'adaptive binned'),
                       (sample, weights, ar, 'weighted adaptive binned')):

        values, edges = hepspt.adbin_hist1d(s, wgts = w, nbins = bins, rg = rg)

        centers = (edges[1:] + edges[:-1])/2.

        xerr = edges[1:] - centers
        yerr = np.sqrt(values)


        a.errorbar(centers, values, yerr, xerr, ls = 'None')
        a.set_ylim(0, 1.5*values.max())
        a.set_ylabel('entries')
        a.set_title(t)

    plt.show()


if __name__ == '__main__':
    '''
    Execute this module as a script.
    '''
    main()
