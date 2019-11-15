'''
Test functions for the "plotting" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__ = ['miguel.ramos.pernas@cern.ch']


# Python

# Local


import matplotlib
import numpy as np
import os
import pytest
import hep_spt


def test_available_styles():
    '''
    Test for the function "available_styles".
    '''
    styles = {'default', 'singleplot', 'multiplot'}
    assert len(set(hep_spt.available_styles()) - styles) == 0


def test_corr_hist2d():
    '''
    Test for the "corr_hist2d" function.
    '''
    matrix = np.array([[1., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.]])
    hep_spt.corr_hist2d(matrix, ['a', 'b', 'c'])


def test_modified_format():
    '''
    Test for the "modified_format" function.
    '''
    prev = matplotlib.rcParams['font.size']
    with hep_spt.modified_format({'font.size': 10}):
        assert matplotlib.rcParams['font.size'] == 10
    assert matplotlib.rcParams['font.size'] == prev


def test_opt_fig_div():
    '''
    Test for the "opt_fig_div" function.
    '''
    assert hep_spt.opt_fig_div(4) == (2, 2)
    assert hep_spt.opt_fig_div(9) == (3, 3)
    assert hep_spt.opt_fig_div(5) == (2, 3)
    assert hep_spt.opt_fig_div(6) == (2, 3)


def test_path_to_styles():
    '''
    Test for the function "path_to_styles".
    '''
    path = hep_spt.path_to_styles()

    s = set(map(lambda s: s[:s.find('.mplstyle')], os.listdir(path)))

    assert len(s - set(hep_spt.available_styles())) == 0


def test_samples_cycler():
    '''
    Test for the function "test_samples_cycler".
    '''
    cfg = {
        'K': 'k',
        'W': 'w',
        'R': 'r',
        'Y': 'y',
        'G': 'g',
        'C': 'c',
        'B': 'b',
        'M': 'm',
    }

    # Construct a cycler
    cyc = hep_spt.samples_cycler(cfg.keys(), ls=cfg.values())
    for c in cyc:
        c['ls'] == cfg[c['label']]

    # Check that a warning is displayed when the number of samples is
    # greater than the number of styles. The check is done considering
    # that the number of samples is a multiple and non-multiple of the
    # number of styles.

    with pytest.warns(RuntimeWarning):

        ls = list(sorted(cfg.values())[:5])
        cyc = hep_spt.samples_cycler(cfg.keys(), ls=ls)

        assert len(cyc) == len(cfg)

        cyc_ls = list(c['ls'] for c in cyc)

        assert ls + ls[:3] == cyc_ls

    with pytest.warns(RuntimeWarning):

        ls = list(sorted(cfg.values())[:4])
        cyc = hep_spt.samples_cycler(cfg.keys(), ls=ls)

        assert len(cyc) == len(cfg)

        cyc_ls = list(c['ls'] for c in cyc)

        assert 2*ls == cyc_ls


def test_set_style():
    '''
    Test for the "set_style" function.
    '''
    for s in hep_spt.available_styles():
        hep_spt.set_style(s)


def test_text_in_rectangles():
    '''
    Test the "text_in_rectangles" function.
    '''
    smp = np.array([
        np.array([0., 0.,  1., 1.]),
        np.array([0., 1.,  0., 1.])
    ]).T
    weights = np.array([2,  1,   2,  1])

    nbins = 2
    bins = hep_spt.adbin_hist(smp, nbins)

    recs, conts = hep_spt.adbin_hist2d_rectangles(bins, smp)

    hep_spt.text_in_rectangles(recs, map(str, conts))
