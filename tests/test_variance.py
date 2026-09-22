"""F 方差齐性检验的对拍测试（scipy 无同名检验，按公式 + R 约定核对）。"""
import numpy as np
import pytest
from scipy import stats

from mystats.hypothesis import ftest_2samp_var

rng = np.random.default_rng(20260922)


def _assert_close(a, b):
    assert np.isclose(a, b, rtol=1e-9, atol=1e-12), f"{a} != {b}"


def test_ftest_two_sided():
    a = rng.normal(0.0, 1.5, size=20)
    b = rng.normal(0.0, 1.0, size=25)
    F = a.var(ddof=1) / b.var(ddof=1)
    dfn, dfd = len(a) - 1, len(b) - 1
    p_ref = 2 * min(stats.f.cdf(F, dfn, dfd), stats.f.sf(F, dfn, dfd))
    res = ftest_2samp_var(a, b)
    _assert_close(res.statistic, F)
    _assert_close(res.pvalue, p_ref)


@pytest.mark.parametrize(
    "alternative,ref_p",
    [
        ("less", lambda F, dfn, dfd: stats.f.cdf(F, dfn, dfd)),
        ("greater", lambda F, dfn, dfd: stats.f.sf(F, dfn, dfd)),
    ],
)
def test_ftest_one_sided(alternative, ref_p):
    a = rng.normal(0.0, 1.2, size=15)
    b = rng.normal(0.0, 1.0, size=22)
    F = a.var(ddof=1) / b.var(ddof=1)
    dfn, dfd = len(a) - 1, len(b) - 1
    res = ftest_2samp_var(a, b, alternative=alternative)
    _assert_close(res.pvalue, ref_p(F, dfn, dfd))


def test_ftest_zero_variance_error():
    with pytest.raises(ValueError):
        ftest_2samp_var([1, 1, 1], [1, 2, 3]) # 一组数据完全相同
