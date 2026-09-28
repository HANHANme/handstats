"""对拍测试：同样的数据，本包的结果必须与 scipy.stats 完全一致。

这是全包正确性的“锚”——每个新检验上线时，都必须在这里补一条对拍，
对上了才有资格进主干。另附一组能手算的小数据（教材式验收）。
"""
import numpy as np
import pytest
from scipy import stats

from mystats.hypothesis import (
    ftest_2samp_var,
    ttest_1samp,
    ttest_2samp_ind,
    ztest_1prop,
    ztest_1samp,
)

rng = np.random.default_rng(20260922)


def _assert_close(a, b):
    assert np.isclose(a, b, rtol=1e-9, atol=1e-12), f"{a} != {b}"


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_ttest_1samp_matches_scipy(alternative):
    x = rng.normal(3.0, 1.5, size=25)
    ref = stats.ttest_1samp(x, 3.0, alternative=alternative)
    res = ttest_1samp(x, 3.0, alternative=alternative)
    _assert_close(res.statistic, ref.statistic)
    _assert_close(res.pvalue, ref.pvalue)


@pytest.mark.parametrize("equal_var", [True, False])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_ttest_2samp_matches_scipy(alternative, equal_var):
    a = rng.normal(0.0, 1.0, size=30)
    b = rng.normal(0.4, 2.0, size=18)
    ref = stats.ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
    res = ttest_2samp_ind(a, b, equal_var=equal_var, alternative=alternative)
    _assert_close(res.statistic, ref.statistic)
    _assert_close(res.pvalue, ref.pvalue)


def test_ztest_1samp_matches_manual_formula():
    x = rng.normal(10.0, 2.0, size=40)
    z_ref = (x.mean() - 10.0) / (2.0 / np.sqrt(len(x)))
    p_ref = 2 * stats.norm.sf(abs(z_ref))
    res = ztest_1samp(x, 10.0, sigma=2.0)
    _assert_close(res.statistic, z_ref)
    _assert_close(res.pvalue, p_ref)


def test_hand_verifiable_numbers():
    """用一组能手算的小数据核对统计量（教材式验收）。

    数据 [1,2,3,4,5]：均值 = 3，s = √2.5 ≈ 1.5811；
    检验 μ0 = 0 时 t = 3/(1.5811/√5) = 4.24264...，p ≈ 0.0134。
    """
    res = ttest_1samp([1.0, 2.0, 3.0, 4.0, 5.0], mu0=3.0)
    _assert_close(res.statistic, 0.0) # 数据正好关于 μ0 对称
    _assert_close(res.pvalue, 1.0)

    res = ttest_1samp([1.0, 2.0, 3.0, 4.0, 5.0], mu0=0.0)
    _assert_close(res.statistic, 4.242640687119285)
    assert res.reject # p ≈ 0.0134 < 0.05


def test_reject_follows_alpha():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert ttest_1samp(x, 0.0, alpha=0.05).reject # p ≈ 0.0134 ≤ 0.05
    assert not ttest_1samp(x, 0.0, alpha=0.005).reject # p > 0.005


def test_one_sided_h0_phrasing_is_composite():
    """单侧检验的 H0 应是复合假设（教材写法），双侧保持等号。"""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    res_g = ttest_1samp(x, 3.0, alternative="greater")
    assert res_g.h0 == "μ ≤ 3" and res_g.h1 == "μ > 3"
    res_l = ttest_1samp(x, 3.0, alternative="less")
    assert res_l.h0 == "μ ≥ 3" and res_l.h1 == "μ < 3"
    res_2 = ttest_1samp(x, 3.0) # 双侧
    assert res_2.h0 == "μ = 3" and res_2.h1 == "μ ≠ 3"
    # 比例与方差检验同样遵循复合写法
    res_p = ztest_1prop(65, 100, 0.5, alternative="less")
    assert res_p.h0 == "p ≥ 0.5" and res_p.h1 == "p < 0.5"
    res_f = ftest_2samp_var([1, 2, 3], [2, 4, 6], alternative="greater")
    assert res_f.h0 == "σ1² ≤ σ2²" and res_f.h1 == "σ1² > σ2²"
    # 双样本均值
    res_2s = ttest_2samp_ind([1, 2, 3], [3, 4, 5], alternative="less")
    assert res_2s.h0 == "μ1 ≥ μ2" and res_2s.h1 == "μ1 < μ2"


def test_one_sided_conclusion_names_direction_on_reject():
    """单侧拒绝时结论应指明方向（数据均值 3 > 1，右侧检验拒绝）。"""
    res = ttest_1samp([1, 2, 3, 4, 5], mu0=1.0, alternative="greater")
    assert res.reject
    text = str(res)
    assert "支持 H1" in text and "μ > 1" in text
    # 双侧结论不添加方向性表述；单侧不拒绝时也不添加
    assert "支持 H1" not in str(ttest_1samp([1, 2, 3, 4, 5], mu0=1.0))
    assert "支持 H1" not in str(ttest_1samp([1, 2, 3, 4, 5], mu0=9.0,
                                            alternative="greater"))
