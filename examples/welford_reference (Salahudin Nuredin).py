"""Compare Welford's results with NumPy's reference implementation."""

import numpy as np

from stateskol.welford_salahudin_nuredin import welford


def main() -> None:
    """Compare Welford statistics with NumPy on a fixed dataset."""
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

    numpy_mean = np.mean(data)
    numpy_variance = np.var(data, ddof=1)
    numpy_std_dev = np.std(data, ddof=1)

    print(f"Data: {data}")
    print(f"Welford mean: {mean}")
    print(f"NumPy mean: {numpy_mean}")
    print(f"Welford sample variance: {variance}")
    print(f"NumPy sample variance: {numpy_variance}")
    print(f"Welford sample standard deviation: {std_dev}")
    print(f"NumPy sample standard deviation: {numpy_std_dev}")
    print(f"Dropped values: {dropped}")


if __name__ == "__main__":
    main()





