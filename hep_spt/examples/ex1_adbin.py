'''
Examples related to the "adbin" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Custom
import hep_spt
hep_spt.set_style()

# Python
import matplotlib.pyplot as plt
import numpy as np


__all__ = ['main']


def main():
    '''
    Main function called when executing this module as a script.
    '''
    size = 1000
    bins = 20
    rg   = (-10, 10)

    sample  = np.random.normal(0., 2, size)
    weights = np.random.uniform(0, 1, size)

    figs, (root, al, ar) = plt.subplots(1, 3, figsize=(15, 5))

    root.hist(sample, bins, rg, label = 'raw')
    root.set_title('raw')

    for s, w, a, t in ((sample, None, al, 'adaptive binned'),
                       (sample, weights, ar, 'weighted adaptive binned')):

        values, edges, ex, ey = hep_spt.adbin_hist1d(s, wgts=w, nbins=bins, rg=rg)
        centers = (edges[1:] + edges[:-1])/2.

        a.errorbar(centers, values, ey, ex, ls = 'None')
        a.set_ylim(0, 1.5*values.max())
        a.set_title(t)

    plt.show()


if __name__ == '__main__':

    main()
