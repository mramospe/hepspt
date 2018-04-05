'''
Test functions for the "cls" module.
'''

__author__ = ['Miguel Ramos Pernas']
__email__  = ['miguel.ramos.pernas@cern.ch']


# Python
import numpy as np
from scipy.stats import poisson, norm
import pytest

# Local
import hep_spt


def test_cls_hypo():
    '''
    Test all the constructor related to the CLsHypo* classes.
    '''
    pm  = 4
    pma = np.arange(pm, pm + 4)

    for m in (pm, pma):
        dh = hep_spt.CLsHypo_discrete(poisson(m))
        dh = hep_spt.cls_hypo(poisson, m)

    gm  = 10
    gma = np.arange(gm, gm + 5)
    gs  = 2
    gsa = np.arange(gs, gs + 5)

    for m, s in ((gm, gs), (gma, gsa)):
        ch = hep_spt.CLsHypo_continuous(norm(m, s))
        ch = hep_spt.cls_hypo(norm, m, s)


def test_cls_ts():
    '''
    Test all the constructors and behaviours of the CLsTS* classes
    '''
    mean  = 8
    sigma = 2

    dh = hep_spt.CLsHypo_discrete(poisson(mean))
    ch = hep_spt.CLsHypo_continuous(norm(mean, sigma))

    with pytest.raises(RuntimeError):
        hep_spt.cls_ts(dh, ch)

    ts = hep_spt.cls_ts(dh, dh)

    res = ts.evaluate(mean)

    assert res.CLs == 1.
