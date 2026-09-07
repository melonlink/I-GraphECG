"""QRS/ST/T time masks are built correctly from the time axis (s), with the R peak at t=0."""
import pytest

torch = pytest.importorskip("torch")

from igraphecg.models.surrogate_decoder import SurrogateDecoder
from igraphecg.training.losses import QRS_WIN, ST_WIN, T_WIN, build_masks


def test_masks_match_windows():
    dec = SurrogateDecoder(n_t=100)
    t = dec.t
    m = build_masks(t)
    for name, win in (("qrs", QRS_WIN), ("st", ST_WIN), ("t", T_WIN)):
        sel = m[name].bool()
        assert sel.sum() > 0
        assert t[sel].min() >= win[0] - 1e-6
        assert t[sel].max() <= win[1] + 1e-6


def test_r_peak_at_zero():
    dec = SurrogateDecoder(n_t=100)
    # R-peak index is the sample closest to t=0 (about 30 for the window [-0.3,0.69])
    idx = int(torch.argmin(dec.t.abs()))
    assert 29 <= idx <= 31
