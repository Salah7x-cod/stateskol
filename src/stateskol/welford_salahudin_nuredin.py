"""Online variance calculation using Welford's algorithm."""

import math
from collections.abc import Iterable
from numbers import Real


def welford(data: Iterable[Real | None]) -> tuple[int, float, float, float, int]:
    """Calculate sample statistics using Welford's online algorithm.

    Parameters
    ----------
    data:
        An iterable of real numeric observations. ``None`` and floating-point
        ``NaN`` values are treated as missing and dropped. Boolean values,
        infinite values, and non-numeric values are invalid. Numeric
        observations are converted to ``float`` before calculation.

    Returns
    -------
    tuple[int, float, float, float, int]
        A tuple containing:
        ``(count, mean, sample_variance, sample_std_dev, dropped_count)``.

        ``count`` is the number of valid observations used in the calculation.
        ``dropped_count`` is the number of missing observations that were
        ignored.

    Raises
    ------
    TypeError
        If an observation is not a real numeric value or is a boolean.
    ValueError
        If an observation is infinite or if fewer than two valid observations
        remain, making sample variance undefined.

    Notes
    -----
    The algorithm maintains only the number of observations, the running
    mean, and the corrected sum of squares (M2). It does not store the input
    observations.

    Missing values are dropped rather than imputed, and the number dropped is
    explicitly returned.

    The sample variance uses ``M2 / (count - 1)`` with Bessel's correction.

    Examples
    --------
    >>> welford([2, 4, 6, 8])
    (4, 5.0, 6.666666666666667, 2.581988897471611, 0)
    """
    count = 0
    mean = 0.0
    m2 = 0.0
    dropped = 0

    for value in data:
        if value is None:
            dropped += 1
            continue

        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"invalid observation {value!r}: expected a real number"
            )

        value = float(value)

        if math.isnan(value):
            dropped += 1
            continue

        if not math.isfinite(value):
            raise ValueError(
                f"invalid observation {value!r}: infinite values are not allowed"
            )

        count += 1

        delta = value - mean
        mean += delta / count
        delta2 = value - mean
        m2 += delta * delta2

    if count < 2:
        raise ValueError(
            "sample variance requires at least two valid observations"
        )

    sample_variance = m2 / (count - 1)
    sample_std_dev = math.sqrt(sample_variance)

    return count, mean, sample_variance, sample_std_dev, dropped