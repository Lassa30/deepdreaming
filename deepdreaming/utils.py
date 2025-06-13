import os


def read_image_net_classes(filepath):
    """Read ImageNet class labels from text file and return as dictionary.

    Args:
        filepath (str): Path to text file containing ImageNet class mappings.
                        Expected format: "index, class_name" per line.

    Returns:
        dict: Dictionary mapping class indices (int) to class names (str).
              Contains exactly 1000 ImageNet classes.

    Raises:
        AssertionError: If filepath doesn't exist or file doesn't contain 1000 classes.
    """
    assert os.path.exists(filepath), f"Not valid filepath: {filepath}"

    result = {}
    with open(filepath, "r") as file:
        for line in file.readlines():
            if line[0] in "0123456789":
                idx, label = map(str.strip, line.split(","))
                result[int(idx)] = label

    assert len(result) == 1000, "Not enough classes in the file"
    return result


def two_max_divisors(n) -> tuple[int, int]:
    """Find two divisors of n that are closest to each other for grid layout.

    Args:
        n (int): Number to find divisors for.

    Returns:
        tuple[int, int]: Two divisors (rows, cols) where rows >= cols and rows * cols = n.
                         Optimized for creating square-like grid layouts.
    """
    for d in range(int(n**0.5) + 1, 0, -1):
        if n % d == 0:
            return n // d, d
    return n, 1


def return_none(func):
    """Decorator that returns None if first argument is None, otherwise calls function normally.

    Args:
        func (callable): Function to decorate.

    Returns:
        callable: Wrapped function that handles None inputs gracefully.
    """
    def wrapper(*args, **kwargs):
        if args[0] is None:
            return None
        return func(*args, **kwargs)

    return wrapper
