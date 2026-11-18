from inspect import signature

from bfm import CaviBFM, GibbsBFM
from bfm._gs import GS_LH
from bfm._vi import CAVI


def test_estimator_and_driver_defaults_agree():
    gs = GibbsBFM(K=2)
    assert gs.c1 == CaviBFM(K=2).c1 == 2.3
    assert signature(GS_LH).parameters["c1"].default == signature(CAVI).parameters["c1"].default == 2.3

    for name, expected in (("n_iter", 5000), ("burn_in", 2500), ("thin", 1)):
        assert getattr(gs, name) == signature(GS_LH).parameters[name].default == expected
