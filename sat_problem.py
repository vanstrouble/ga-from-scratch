import os
import random
# import numpy as np


def get_random_filename(path="data/"):
    files = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isfile(os.path.join(path, name))
    ]

    if not files:
        raise FileNotFoundError(f"No files found in '{path}'")

    return random.choice(files)


def read_sat_instance(filename):
    with open(filename, "r") as f:
        pass


if __name__ == "__main__":
    filename = get_random_filename()
    print(f"Selected file: {filename}")
