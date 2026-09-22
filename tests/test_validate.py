"""输入校验的单元测试。"""
import numpy as np
import pytest

from mystats.validate import (
    as_sample,
    check_alpha,
    check_alternative,
    check_confidence,
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
