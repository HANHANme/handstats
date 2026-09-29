"""区间估计的对拍测试：与 scipy.stats 的 interval 函数核对。"""
import numpy as np
from scipy import stats

from handstats.interval import ci_mean

rng = np.random.default_rng(20260922)


def test_ci_mean_t_matches_scipy():
    x = rng.normal(2.0, 1.0, size=18)
    n = len(x)
    ref_lo, ref_hi = stats.t.interval(
        0.95, n - 1, loc=x.mean(), scale=x.std(ddof=1) / np.sqrt(n)
    )
    res = ci_mean(x, confidence=0.95)
    assert np.isclose(res.lower, ref_lo, rtol=1e-9)
    assert np.isclose(res.upper, ref_hi, rtol=1e-9)


def test_ci_mean_z_matches_scipy():
    x = rng.normal(7.0, 2.0, size=12)
    sigma = 2.0
    ref_lo, ref_hi = stats.norm.interval(
        0.90, loc=x.mean(), scale=sigma / np.sqrt(len(x))
    )
    res = ci_mean(x, confidence=0.90, sigma=sigma)
    assert np.isclose(res.lower, ref_lo, rtol=1e-9)
    assert np.isclose(res.upper, ref_hi, rtol=1e-9)


def test_contains_and_str():
    res = ci_mean([1.0, 2.0, 3.0, 4.0, 5.0])
    assert res.contains(3.0)
    assert "置信区间" in str(res)
