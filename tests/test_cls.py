'''
Test functions for the "cls" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__ = ['miguel.ramos.pernas@cern.ch']

import hep_spt
import numpy as np
import pytest
from scipy.stats import poisson, norm


def test_cls_hypo():
    '''
    Test all the constructor related to the CLsHypo* classes.
    '''
    pm = 4
    pma = np.arange(pm, pm + 4)

    for m in (pm, pma):
        dh = hep_spt.CLsHypo_discrete(poisson(m))
        dh = hep_spt.cls_hypo(poisson, m)

    gm = 10
    gma = np.arange(gm, gm + 5)
    gs = 2
    gsa = np.arange(gs, gs + 5)

    for m, s in ((gm, gs), (gma, gsa)):
        ch = hep_spt.CLsHypo_continuous(norm(m, s))
        ch = hep_spt.cls_hypo(norm, m, s)


def test_clshypo():
    '''
    Test for the "CLsHypo" base class.
    '''
    h = hep_spt.CLsHypo(norm(0, 2))

    assert np.allclose(h.median(), 0)
    assert np.allclose(h.percentil(0.5), 0)


def test_clshypo_continuous():
    '''
    Test for the "CLsHypo_continuous" class.
    '''
    h = hep_spt.CLsHypo_continuous(norm(8, 2))

    assert np.allclose(h(8), norm(8, 2).pdf(8))


def test_clshypo_discrete():
    '''
    Test for the "CLsHypo_discrete" class.
    '''
    h = hep_spt.CLsHypo_discrete(poisson(8))

    assert np.allclose(h(8), poisson(8).pmf(8))


def test_cls_ts():
    '''
    Test for the "cls_ts" function.
    '''
    mean = 8
    sigma = 2

    dh = hep_spt.CLsHypo_discrete(poisson(mean))
    ch = hep_spt.CLsHypo_continuous(norm(mean, sigma))

    with pytest.raises(RuntimeError):
        hep_spt.cls_ts(dh, ch)

    ts = hep_spt.cls_ts(dh, dh)

    res = ts.evaluate(mean)

    assert res.CLs == 1.


def test_clsts():
    '''
    Test for the "CLsTS" base class.
    '''
    alt = hep_spt.CLsHypo_discrete(poisson(8))
    null = hep_spt.CLsHypo_discrete(poisson(4))

    hep_spt.CLsTS(alt, null)


def test_clsts_continuous():
    '''
    Test for the "CLsTS_continuous" class.
    '''
    alt = hep_spt.CLsHypo_continuous(norm(8, 2))
    null = hep_spt.CLsHypo_continuous(norm(4, 2))

    h = hep_spt.CLsTS_continuous(alt, null)


def test_clsts_discrete():
    '''
    Test for the "CLsTS_discrete" class.
    '''
    alt = hep_spt.CLsHypo_discrete(poisson(8))
    null = hep_spt.CLsHypo_discrete(poisson(4))

    h = hep_spt.CLsTS_discrete(alt, null)
