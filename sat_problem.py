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
    chromosome_length = None
    n_clauses = None
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("p cnf"):
                parts = line.split()
                chromosome_length = int(parts[2])
                n_clauses = int(parts[3])
                break

        if chromosome_length is None or n_clauses is None:
            raise ValueError("Invalid CNF file format: missing 'p cnf' line")

        clauses = []
        while len(clauses) < n_clauses:
            clause_line = f.readline().strip()
            if not clause_line or clause_line.startswith("c"):
                continue
            clause = list(map(int, clause_line.split()[:-1]))
            clauses.append(clause)

    return chromosome_length, n_clauses, clauses


if __name__ == "__main__":
    filename = get_random_filename()
    print(f"Selected file: {filename}")
    chromosome_length, n_clauses, clauses = read_sat_instance(filename)
    print(f"Chromosome length: {chromosome_length}")
    print(f"Number of clauses: {n_clauses}")
    print(f"Clauses shape: {len(clauses)} x {len(clauses[0])}")
    print(f"Clauses sample: {clauses[:5]}")
