'''
Examples related to the "plotting" module.
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
    smp  = np.random.normal(0, 3, size)
    wgts = np.random.uniform(0, 1, size)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))

    # Make a histogram with poissonian errors
    centers, values, ey, ex = hep_spt.errorbar_hist(smp, rg=(-7, 7))

    ax0.errorbar(centers, values, ey, ex, ls = 'none')
    ax0.set_title('Non-weighted sample')

    # Make a weighted histogram (with square of sum of weights errors)
    centers, values, ey, ex = hep_spt.errorbar_hist(smp, rg=(-7, 7), wgts=wgts)

    ax1.errorbar(centers, values, ey, ex, ls = 'none')
    ax1.set_title('Weighted sample')

    # Show the histograms
    plt.show()


if __name__ == '__main__':

    main()
