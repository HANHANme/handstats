"""输入校验的单元测试。"""
import numpy as np
import pytest

from handstats.validate import (
    as_counts,
    as_sample,
    as_table,
    check_alpha,
    check_alternative,
    check_confidence,
    check_proportion,
    check_successes,
)


def test_as_sample_converts_and_checks():
    arr = as_sample([1, 2, 3])
    assert arr.dtype == np.float64
    assert arr.shape == (3,)


def test_as_sample_rejects_bad_input():
    with pytest.raises(ValueError):
        as_sample([[1, 2], [3, 4]]) # 二维
    with pytest.raises(ValueError):
        as_sample([]) # 空
    with pytest.raises(ValueError):
        as_sample([1.0, np.nan, 3.0]) # 含 NaN


def test_alternative_aliases():
    assert check_alternative("two-sided") == "two-sided"
    assert check_alternative("双侧") == "two-sided"
    assert check_alternative("Less") == "less"
    assert check_alternative("右侧") == "greater"
    with pytest.raises(ValueError):
        check_alternative("both")


def test_alpha_and_confidence_bounds():
    with pytest.raises(ValueError):
        check_alpha(0)
    with pytest.raises(ValueError):
        check_alpha(1)
    with pytest.raises(ValueError):
        check_confidence(1.5)


def test_as_counts_rejects_bad_frequencies():
    with pytest.raises(ValueError):
        as_counts([5, -1, 3]) # 负频数
    with pytest.raises(ValueError):
        as_counts([0, 0]) # 总频数为 0


def test_as_table_rejects_bad_tables():
    with pytest.raises(ValueError):
        as_table([1, 2, 3]) # 一维
    with pytest.raises(ValueError):
        as_table([[5, 1], [2, -3]]) # 负频数
    with pytest.raises(ValueError):
        as_table([[3]]) # 不是 2×2


def test_check_proportion_and_successes():
    assert check_proportion(0.5, "p0") == 0.5
    assert check_successes(65, 100, "x") == 65.0
    with pytest.raises(ValueError):
        check_proportion(1.5, "p0")
    with pytest.raises(ValueError):
        check_successes(101, 100, "x") # x > n
    with pytest.raises(ValueError):
        check_successes(50.5, 100, "x") # 非整数
