import math
import numpy as np
import pytest

from stateskol.welford_salahudin_nuredin import welford


def test_welford_small_dataset():
    count, mean, variance, std_dev, dropped = welford([2, 4, 6, 8])

    assert count == 4
    assert mean == pytest.approx(5.0)
    assert variance == pytest.approx(20 / 3)
    assert std_dev == pytest.approx(math.sqrt(20 / 3))
    assert dropped == 0


def test_welford_single_valid_value_fails():
    with pytest.raises(ValueError, match="at least two"):
        welford([5])


def test_welford_empty_input_fails():
    with pytest.raises(ValueError, match="at least two"):
        welford([])


def test_welford_constant_values():
    count, mean, variance, std_dev, dropped = welford([7, 7, 7, 7])

    assert count == 4
    assert mean == pytest.approx(7.0)
    assert variance == pytest.approx(0.0)
    assert std_dev == pytest.approx(0.0)
    assert dropped == 0


def test_welford_drops_none_and_nan():
    count, mean, variance, std_dev, dropped = welford(
        [1, None, 2, float("nan"), 3]
    )

    assert count == 3
    assert mean == pytest.approx(2.0)
    assert variance == pytest.approx(1.0)
    assert std_dev == pytest.approx(1.0)
    assert dropped == 2


def test_welford_rejects_non_numeric_values():
    with pytest.raises(TypeError, match="expected a real number"):
        welford([1, 2, "three"])


def test_welford_rejects_infinite_values():
    with pytest.raises(ValueError, match="infinite"):
        welford([1, 2, float("inf")])


def test_welford_negative_values():
    count, mean, variance, std_dev, dropped = welford([-2, -4, -6, -8])

    assert count == 4
    assert mean == pytest.approx(-5.0)
    assert variance == pytest.approx(20 / 3)
    assert std_dev == pytest.approx(math.sqrt(20 / 3))
    assert dropped == 0


def test_welford_accepts_generators():
    data = (value for value in [2, 4, 6, 8])

    count, mean, variance, std_dev, dropped = welford(data)

    assert count == 4
    assert mean == pytest.approx(5.0)
    assert variance == pytest.approx(20 / 3)
    assert std_dev == pytest.approx(math.sqrt(20 / 3))
    assert dropped == 0

def test_welford_matches_numpy_on_reference_dataset():
    data = [
        12.5,
        8.25,
        17.75,
        3.5,
        21.0,
        14.25,
        9.75,
        18.5,
    ]

    count, mean, variance, std_dev, dropped = welford(data)

    assert count == len(data)
    assert dropped == 0
    assert mean == pytest.approx(np.mean(data), rel=1e-12, abs=1e-12)
    assert variance == pytest.approx(
        np.var(data, ddof=1),
        rel=1e-12,
        abs=1e-12,
    )
    assert std_dev == pytest.approx(
        np.std(data, ddof=1),
        rel=1e-12,
        abs=1e-12,
    )

def test_welford_is_stable_for_large_offset_data():
    data = [
        1_000_000_000_001,
        1_000_000_000_002,
        1_000_000_000_003,
        1_000_000_000_004,
    ]

    count, mean, variance, std_dev, dropped = welford(data)

    assert count == 4
    assert dropped == 0
    assert mean == pytest.approx(1_000_000_000_002.5)
    assert variance == pytest.approx(5 / 3)
    assert std_dev == pytest.approx(math.sqrt(5 / 3))

def test_welford_fails_fast_on_invalid_value():
    with pytest.raises(TypeError):
        welford([1, 2, "invalid", 4])
