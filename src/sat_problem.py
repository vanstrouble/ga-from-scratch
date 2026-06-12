import os
import random
import numpy as np
import matplotlib.pyplot as plt
import genetic_algorithm as GA
from multiprocessing import Pool, cpu_count


def get_random_filename(path="data/", n_files=1):
    """
    Get a random filename from the specified directory.
    Args:
        path (str): The directory to search for files.
        n_files (int): The number of files to select. Must be a positive integer.
    Returns:
        list: A list of paths to randomly selected files.
    """
    if n_files < 1:
        raise ValueError("n_files must be a positive integer")

    files = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isfile(os.path.join(path, name))
    ]

    if not files:
        raise FileNotFoundError(f"No files found in '{path}'")
    if n_files > len(files):
        raise ValueError(f"Requested {n_files} files, but only {len(files)} available in '{path}'")

    return random.sample(files, n_files)


def read_sat_instance(filename):
    """
    Read a SAT instance from a CNF file.
    Args:
        filename (str): The path to the CNF file.
    Returns:
        tuple: A tuple containing the chromosome length,
        number of clauses, and the clauses themselves.
    """
    chromosome_length = None
    n_clauses = None
    clause_length = None
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("c"):
                if "clause length = " in line:
                    clause_length = int(line.split("=")[1].strip())
                continue
            if line.startswith("p cnf"):
                parts = line.split()
                chromosome_length = int(parts[2])
                n_clauses = int(parts[3])
                break

        if chromosome_length is None or n_clauses is None:
            raise ValueError("Invalid CNF file format: missing 'p cnf' line")
        if clause_length is None:
            clause_length = 3

        clauses = np.empty((n_clauses, clause_length), dtype=int)
        clause_index = 0

        while clause_index < n_clauses:
            clause_line = f.readline().strip()
            if not clause_line or clause_line.startswith("c"):
                continue
            clause = list(map(int, clause_line.split()[:-1]))
            if len(clause) != clause_length:
                continue
            clauses[clause_index] = clause
            clause_index += 1

    return chromosome_length, n_clauses, clauses


def generate_sat_fitness(clauses):
    """
    Generate a fitness function for the SAT problem based on the given clauses.
    Args:
        clauses (np.ndarray): A 2D array where each row represents a clause
        and each element is a literal (positive for true, negative for false).
    Returns:
        function: A fitness function that takes a chromosome and returns
        the number of satisfied clauses.
    """
    idxs = np.abs(clauses) - 1
    positive_mask = clauses > 0

    def fitness_func(chromosome):
        """
        Calculate the fitness of a chromosome based on the number of satisfied clauses.
        Args:
            chromosome (np.ndarray): A binary array representing a solution.
        Returns:
            int: The number of satisfied clauses.
        """
        gene_values = chromosome[idxs]
        satisfied_literals = (gene_values == positive_mask)
        satisfied_clauses = np.any(satisfied_literals, axis=1)

        return np.sum(satisfied_clauses, dtype=np.int64)

    return fitness_func


def evaluate_instance(args):
    """
    Evaluate a single SAT instance using the genetic algorithm.
    Args:
        args (tuple): A tuple containing the filename, pop_size, crossover_rate,
        elitism_rate, and iter_num.
    Returns:
        dict: Result dictionary with file, best_fitness, n_clauses, and satisfaction.
    """
    filename, pop_size, crossover_rate, elitism_rate, iter_num = args
    chromosome_length, n_clauses, clauses = read_sat_instance(filename)
    fitness_func = generate_sat_fitness(clauses)

    ga = GA.GA(
        pop_size=pop_size,
        str_size=chromosome_length,
        fitness_func=fitness_func,
        mutation_rate=1 / chromosome_length,
        crossover_rate=crossover_rate,
        elitism_rate=elitism_rate,
    )
    ga.run(iter_num=iter_num)

    satisfaction = ga.best_fitness / n_clauses * 100
    return {
        "file": filename,
        "best_fitness": ga.best_fitness,
        "n_clauses": n_clauses,
        "satisfaction": satisfaction,
    }


if __name__ == "__main__":
    # Test GA on a single SAT instance
    filename = get_random_filename()
    if isinstance(filename, (list, tuple)):
        filename = filename[0]
    print(f"Selected file: {filename}")

    chromosome_length, n_clauses, clauses = read_sat_instance(filename)
    print(f"Chromosome length: {chromosome_length}")
    print(f"Clauses shape: {len(clauses)} x {len(clauses[0])}")
    print(f"Clauses sample: \n{clauses[:5]}")

    fitness_func = generate_sat_fitness(clauses)

    # Hyperparameters
    POP_SIZE = 200
    MUTATION_RATE = 1 / chromosome_length  # Heuristic for mutation rate
    CROSSOVER_RATE = 0.8
    ELITISM_RATE = 0.02
    ITER_NUM = 500

    ga = GA.GA(
        pop_size=POP_SIZE,
        str_size=chromosome_length,
        fitness_func=fitness_func,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        elitism_rate=ELITISM_RATE,
    )

    history, best_solution = ga.run(iter_num=ITER_NUM)

    # Results
    print(f"Best fitness: {ga.best_fitness} / {n_clauses}")
    print(f"Satisfies: {ga.best_fitness / n_clauses * 100:.1f}%")
    print(f"Best solution: {best_solution}")

    # ga.plot_evolution(history, ITER_NUM, problem_name=f"3-SAT Problem ({n_clauses} clauses)")

    # Test GA consistency on the same SAT instance 30 times in parallel
    print("\n\nRunning 30 GA trials on the same SAT instance...\n")
    repeated_runs = 30
    repeated_args = [
        (filename, POP_SIZE, CROSSOVER_RATE, ELITISM_RATE, ITER_NUM)
        for _ in range(repeated_runs)
    ]

    with Pool(processes=cpu_count()) as pool:
        repeated_results = pool.map(evaluate_instance, repeated_args)

    for count, result in enumerate(repeated_results, start=1):
        print(
            f"{count:2d}. {result['file']:10s} • "
            f"{result['best_fitness']}/{result['n_clauses']} "
            f"({result['satisfaction']:.1f}%)"
        )

    repeated_satisfactions = np.array([r["satisfaction"] for r in repeated_results])

    print(f"\nAverage satisfaction: {np.mean(repeated_satisfactions):.2f}%")
    print(f"Standard deviation: {np.std(repeated_satisfactions):.2f}%")

    mean_repeated = float(np.mean(repeated_satisfactions))

    plt.figure(figsize=(10, 5))
    plt.hist(
        repeated_satisfactions,
        bins=min(10, max(1, len(np.unique(repeated_satisfactions)))),
        edgecolor="black",
        alpha=0.85,
    )
    plt.axvline(
        mean_repeated,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label="Mean",
    )
    plt.title(f"SAT satisfaction histogram over {repeated_runs} runs")
    plt.xlabel("Satisfaction (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Test GA generalization on multiple SAT instances
    print("\n\nRunning multiple SAT instances...\n")
    filenames = get_random_filename(n_files=30)

    # Hyperparameters
    POP_SIZE = 200
    CROSSOVER_RATE = 0.8
    ELITISM_RATE = 0.02
    ITER_NUM = 500

    # Run GA on multiple instances in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(
            evaluate_instance,
            [
                (filename, POP_SIZE, CROSSOVER_RATE, ELITISM_RATE, ITER_NUM)
                for filename in filenames
            ],
        )

    for count, result in enumerate(results, start=1):
        print(
            f"{count:2d}. {result['file']:10s} • "
            f"{result['best_fitness']}/{result['n_clauses']} "
            f"({result['satisfaction']:.1f}%)"
        )

    satisfactions = np.array([r["satisfaction"] for r in results])

    print(f"\nAverage satisfaction: {np.mean(satisfactions):.2f}%")
    print(f"Standard deviation: {np.std(satisfactions):.2f}%")
    print(f"Solved instances: {np.sum(satisfactions == 100)} / {len(results)}")
