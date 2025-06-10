import os


def read_image_net_classes(filepath):
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
    for d in range(int(n**0.5) + 1, 0, -1):
        if n % d == 0:
            return n // d, d
    return n, 1


def return_none(func):
    def wrapper(*args, **kwargs):
        if args[0] is None:
            return None
        return func(*args, **kwargs)

    return wrapper
