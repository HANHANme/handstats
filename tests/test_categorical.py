"""分类数据检验的对拍测试：与 scipy.stats 核对 + 手算数据验收。"""
import numpy as np
import pytest
from scipy import stats

from handstats.hypothesis import chisquare_gof, chisquare_ind


def _assert_close(a, b):
    assert np.isclose(a, b, rtol=1e-9, atol=1e-12), f"{a} != {b}"


def test_gof_uniform_matches_scipy():
    obs = [18, 22, 16, 14, 25, 25]
    ref = stats.chisquare(obs) # 缺省均匀分布
    res = chisquare_gof(obs)
    _assert_close(res.statistic, ref.statistic)
    _assert_close(res.pvalue, ref.pvalue)
    _assert_close(res.params["df"], 5)


def test_gof_with_probs_matches_scipy():
    obs = [30, 15, 12, 43]
    probs = [0.4, 0.2, 0.2, 0.2]
    # scipy 的 f_exp 接收“理论频数”，概率要先乘以总频数
    ref = stats.chisquare(obs, f_exp=np.asarray(probs) * sum(obs))
    res = chisquare_gof(obs, expected=probs)
    _assert_close(res.statistic, ref.statistic)
    _assert_close(res.pvalue, ref.pvalue)


def test_gof_with_nparams_matches_scipy():
    obs = [10, 12, 9, 8, 11, 14]
    ref = stats.chisquare(obs, ddof=1) # df = k - 1 - 1
    res = chisquare_gof(obs, n_params=1)
    _assert_close(res.statistic, ref.statistic)
    _assert_close(res.pvalue, ref.pvalue)


def test_gof_hand_verifiable_dice():
    """60 次掷骰子 [8,13,9,12,7,11]：E = 10，χ² = 28/10 = 2.8，df = 5（能手算）。"""
    res = chisquare_gof([8, 13, 9, 12, 7, 11])
    _assert_close(res.statistic, 2.8)
    _assert_close(res.pvalue, stats.chi2.sf(2.8, 5))
    assert not res.reject


def test_gof_accepts_probs_or_counts():
    """expected 给概率（和为 1）或给理论频数（和为 n），结果一致。"""
    res1 = chisquare_gof([30, 15, 12, 43], expected=[0.4, 0.2, 0.2, 0.2])
    res2 = chisquare_gof([30, 15, 12, 43], expected=[40, 20, 20, 20])
    _assert_close(res1.statistic, res2.statistic)
    _assert_close(res1.pvalue, res2.pvalue)


def test_gof_input_errors():
    with pytest.raises(ValueError):
        chisquare_gof([5, -1, 3]) # 负频数
    with pytest.raises(ValueError):
        chisquare_gof([5, 5], expected=[0.5, 0.7, 0.2]) # 长度不一致
    with pytest.raises(ValueError):
        chisquare_gof([5, 5], n_params=2) # df < 1


def test_ind_matches_scipy_2x3():
    table = [[10, 20, 30], [6, 25, 29]]
    ref = stats.chi2_contingency(table, correction=False) # 教材版无 Yates 校正
    res = chisquare_ind(table)
    _assert_close(res.statistic, ref.statistic)
    _assert_close(res.pvalue, ref.pvalue)
    _assert_close(res.params["df"], ref.dof)


def test_ind_matches_scipy_3x3():
    table = [[20, 30, 10], [12, 28, 22], [18, 24, 26]]
    ref = stats.chi2_contingency(table, correction=False)
    res = chisquare_ind(table)
    _assert_close(res.statistic, ref.statistic)
    _assert_close(res.pvalue, ref.pvalue)


def test_ind_hand_verifiable_2x2():
    """[[7,11],[8,19]]：E = [[6,12],[9,18]]，χ² = 15/36 ≈ 0.41667（能手算）。"""
    res = chisquare_ind([[7, 11], [8, 19]])
    _assert_close(res.statistic, 15 / 36)
    _assert_close(res.params["df"], 1)
    assert np.allclose(res.params["expected"], [[6, 12], [9, 18]])


def test_ind_input_errors():
    with pytest.raises(ValueError):
        chisquare_ind([1, 2, 3]) # 一维不是表
    with pytest.raises(ValueError):
        chisquare_ind([[1, 2], [3, -4]]) # 负频数
