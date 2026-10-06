"""Demonstrate Welford's online variance calculation."""

from stateskol.welford_salahudin_nuredin import welford


def main() -> None:
    """Run a small, reproducible Welford example."""
    data = [2, 4, 6, 8]

    count, mean, variance, std_dev, dropped = welford(data)

    print(f"Data: {data}")
    print(f"Count: {count}")
    print(f"Mean: {mean}")
    print(f"Sample variance: {variance}")
    print(f"Sample standard deviation: {std_dev}")
    print(f"Dropped values: {dropped}")


if __name__ == "__main__":
    main()
