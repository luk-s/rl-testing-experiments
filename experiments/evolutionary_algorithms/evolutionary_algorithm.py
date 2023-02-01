import argparse
import logging
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import numpy as np
import yaml

import wandb
from rl_testing.config_parsers import get_engine_config
from rl_testing.engine_generators import get_engine_generator
from rl_testing.evolutionary_algorithms.crossovers import (
    Crossover,
    crossover_half_board,
    crossover_one_eighth_board,
    crossover_one_quarter_board,
)
from rl_testing.evolutionary_algorithms.fitness import (
    EditDistanceFitness,
    Fitness,
    PieceNumberFitness,
)
from rl_testing.evolutionary_algorithms.individuals import BoardIndividual
from rl_testing.evolutionary_algorithms.mutations import (
    Mutator,
    mutate_add_one_piece,
    mutate_castling_rights,
    mutate_flip_board,
    mutate_move_one_piece,
    mutate_move_one_piece_adjacent,
    mutate_move_one_piece_legal,
    mutate_player_to_move,
    mutate_remove_one_piece,
    mutate_rotate_board,
)
from rl_testing.evolutionary_algorithms.selections import Selector, select_tournament
from rl_testing.util.experiment import store_experiment_params

RESULT_DIR = Path(__file__).parent.parent / Path("results/evolutionary_algorithm")
WANDB_CONFIG_FILE = Path(__file__).parent.parent / Path(
    "configs/hyperparameter_tuning_configs/config_ea_edit_distance.yaml"
)
DEBUG = False
DEBUG_CONFIG = {
    "crossover_prob": 0.3379889108848678,
    "mutation_prob": 0.9247393367296812,
    "num_generations": 29,
    "num_runs_per_config": 15,
    "num_workers": 8,
    "population_size": 777,
    "tournament_fraction": 0.6969565124339143,
}

BestFitnessValue = float
WorstFitnessValue = float
AverageFitnessValue = float
UniqueIndividualFraction = float


class FakeConfig:
    def __init__(self, config: Dict[str, Any]) -> None:
        # Assert that all keys of the config dictionary are strings
        assert all(
            [isinstance(key, str) for key in config]
        ), "All dictionary keys must be strings!"

        for key in config:
            setattr(self, key, config[key])


def get_random_state(random_state: Optional[np.random.Generator] = None) -> np.random.Generator:
    """Get a random state. Use the provided random state if it is not None, otherwise use the default random state.

    Args:
        random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        np.random.Generator: The random state.
    """
    if random_state is None:
        return np.random.default_rng()
    return random_state


def get_random_individuals(
    file_path: str, amount: int, _random_state: Optional[np.random.Generator]
) -> List[BoardIndividual]:
    """Get a number of random individuals.

    Args:
        file_path (str): The path to the file containing the boards.
        amount (int): The number of boards to get. It must hold that 0 < amount <= 100_000.
        _random_state (Optional[np.random.Generator]): The random state to use. Defaults to None.

    Returns:
        List[BoardIndividual]: The random boards.
    """
    assert 0 < amount <= 100_000, f"Amount must be between 0 and 100_000, got {amount}."

    random_state = get_random_state(_random_state)

    # Read the fen-strings from the provided file.
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Randomly choose 'amount' fen strings from the file
    fens = random_state.choice(lines, size=amount, replace=False)

    # Convert the fen strings to boards
    individuals = [BoardIndividual(fen) for fen in fens]

    return individuals


def setup_operators(
    random_state: np.random.Generator, population_size: int, tournament_fraction: float
) -> Tuple[Fitness, Mutator, Crossover, Selector]:
    # Initialize the fitness function
    fitness = EditDistanceFitness("3r3k/7p/2p1np2/4p1p1/1Pq1P3/2Q2P2/P4RNP/2R4K b - - 0 42")

    # Initialize the mutation functions
    mutate = Mutator(
        mutation_strategy="all",
        _random_state=random_state,
    )
    mutate.register_mutation_function(
        [
            mutate_player_to_move,
            mutate_add_one_piece,
            mutate_move_one_piece,
            mutate_move_one_piece_adjacent,
            mutate_move_one_piece_legal,
            mutate_remove_one_piece,
            mutate_flip_board,
            mutate_rotate_board,
        ],
        probability=0.2,
        retries=5,
        clear_fitness_values=True,
    )
    mutate.register_mutation_function(
        [
            mutate_castling_rights,
        ],
        probability=0.2,
        probability_per_direction=0.5,
        retries=5,
        clear_fitness_values=True,
    )

    # Initialize the crossover functions
    crossover = Crossover(
        crossover_strategy="all",
        _random_state=random_state,
    )
    crossover.register_crossover_function(
        [
            crossover_half_board,
            crossover_one_quarter_board,
            crossover_one_eighth_board,
        ],
        probability=0.3,
        retries=5,
        clear_fitness_values=True,
    )

    # Initialize the selection functions
    select = Selector(
        selection_strategy="all",
        _random_state=random_state,
    )
    select.register_selection_function(
        select_tournament,
        rounds=population_size,
        tournament_size=int(population_size * tournament_fraction),
        find_best_individual=fitness.best_individual,
    )

    return fitness, mutate, crossover, select


def evolutionary_algorithm(
    population_size: int,
    crossover_prob: float,
    mutation_prob: float,
    num_generations: int,
    tournament_fraction: float,
    num_workers: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[
    List[BoardIndividual],
    List[BestFitnessValue],
    List[AverageFitnessValue],
    List[WorstFitnessValue],
    List[UniqueIndividualFraction],
]:
    for parameter in [
        crossover_prob,
        mutation_prob,
        tournament_fraction,
    ]:
        assert 0 <= parameter <= 1, f"Parameter {parameter = } must be between 0 and 1."

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    random_state = np.random.default_rng(seed)

    # population_size = 1000
    # crossover_prob = 0.2
    # mutation_prob = 0.5
    # n_generations = 50

    # Initialize the operators
    fitness, mutate, crossover, select = setup_operators(
        random_state, population_size, tournament_fraction
    )

    pool = multiprocessing.Pool(processes=num_workers)

    best_fitness_values = []
    average_fitness_values = []
    worst_fitness_values = []
    unique_individual_fractions = []

    # Create the population
    population = get_random_individuals("data/random_positions.txt", population_size, random_state)

    # Evaluate the entire population
    fitness_values = map(fitness.evaluate, population)
    for individual, fitness_val in zip(population, fitness_values):
        individual.fitness = fitness_val

    for generation in range(num_generations):
        if DEBUG:
            print(f"{generation = }")
        # Select the next generation individuals
        offspring = select(population)

        # Clone the selected individuals
        offspring: List[BoardIndividual] = list(map(BoardIndividual.copy, offspring))

        # Apply crossover on the offspring
        mated_children: List[BoardIndividual] = []
        couple_candidates = list(zip(offspring[::2], offspring[1::2]))
        random_values = random_state.random(size=len(offspring) // 2)

        # Filter out the individuals that will mate
        mating_candidates = [
            couple_candidates[i]
            for i, random_value in enumerate(random_values)
            if random_value < crossover_prob
        ]
        single_children = [
            couple_candidates[i]
            for i, random_value in enumerate(random_values)
            if random_value >= crossover_prob
        ]

        # Apply crossover on the mating candidates
        mated_tuples = pool.map(crossover, mating_candidates)
        for individual1, individual2 in mated_tuples + single_children:
            mated_children.append(individual1)
            mated_children.append(individual2)

        # Apply mutation on the offspring
        mutated_children: List[BoardIndividual] = []
        random_values = random_state.random(size=len(mated_children))

        # Filter out the individuals that will mutate
        mutation_candidates = [
            mated_children[i]
            for i, random_value in enumerate(random_values)
            if random_value < mutation_prob
        ]
        non_mutation_candidates = [
            mated_children[i]
            for i, random_value in enumerate(random_values)
            if random_value >= mutation_prob
        ]

        # Apply mutation on the mutation candidates
        mutated_children = pool.map(mutate, mutation_candidates) + non_mutation_candidates

        # Evaluate the individuals with an invalid fitness
        unevaluated_individuals = [
            individual for individual in mutated_children if individual.fitness is None
        ]
        fitness_values = pool.map(fitness.evaluate, unevaluated_individuals)
        for individual, fitness_val in zip(unevaluated_individuals, fitness_values):
            individual.fitness = fitness_val

        # The population is entirely replaced by the offspring
        population = mutated_children

        # Print the best individual and its fitness
        best_individual, best_fitness = fitness.best_individual(population)
        best_fitness_values.append(best_fitness)
        if DEBUG:
            print(f"{best_individual = }, {best_fitness = }")

        # Print the worst individual and its fitness
        worst_individual, worst_fitness = fitness.worst_individual(population)
        worst_fitness_values.append(worst_fitness)
        if DEBUG:
            print(f"{worst_individual = }, {worst_fitness = }")

        # Print the average fitness
        average_fitness = sum(individual.fitness for individual in population) / population_size
        average_fitness_values.append(average_fitness)
        if DEBUG:
            print(f"{average_fitness = }")

        # Print the number of unique individuals
        unique_individuals = set([p.fen() for p in population])
        unique_individual_fractions.append(len(unique_individuals) / population_size)
        if DEBUG:
            print(f"Number of unique individuals = {len(unique_individuals)}")
            print()

    pool.close()

    return (
        population,
        best_fitness_values,
        average_fitness_values,
        worst_fitness_values,
        unique_individual_fractions,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################
    #           CONFIG START         #
    ##################################
    # NETWORKS:
    # =========
    # strong and recent: "network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42"
    # from leela paper: "network_c8368caaccd43323cc513465fb92740ea6d10b50684639a425fca2b42fc1f7be"
    # strong recommended: "network_600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2" # noqa: E501
    # Weak local 1: "f21ee51844a7548c004a1689eacd8b4cd4c6150d6e03c732b211cf9963d076e1"
    # Weak local 2: "fbd5e1c049d5a46c098f0f7f12e79e3fb82a7a6cd1c9d1d0894d0aae2865826f"

    # fmt: off
    # Engine parameters
    parser.add_argument("--seed",               type=int,  default=42)
    parser.add_argument("--engine_config_name", type=str,  default="remote_full_logs_400_nodes.ini")  # noqa: E501
    parser.add_argument("--network_path1",      type=str,  default="T807785-b124efddc27559564d6464ba3d213a8279b7bd35b1cbfcf9c842ae8053721207")  # noqa: E501
    parser.add_argument("--network_path2",      type=str,  default="T785469-600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2")  # noqa: E501
    parser.add_argument("--result_subdir",      type=str,  default="main_results")

    # Evolutionary algorithm parameters

    # fmt: on
    ##################################
    #           CONFIG END           #
    ##################################

    # Set up the logger
    logging.basicConfig(
        format="▸ %(asctime)s.%(msecs)03d %(filename)s:%(lineno)d %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()

    # Parse the arguments
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Get the engine config and engine generator
    engine_config = get_engine_config(
        config_name=args.engine_config_name,
        config_folder_path=Path(__file__).parent.parent.absolute()
        / Path("configs/engine_configs/"),
    )
    engine_generator = get_engine_generator(engine_config)

    # Create results-file-name
    engine_config_name = args.engine_config_name[:-4]
    network_name1 = args.network_path1.split("-")[0]
    network_name2 = args.network_path2.split("-")[0]

    # Store the experiment config in the results file
    result_directory = RESULT_DIR / args.result_subdir
    result_directory.mkdir(parents=True, exist_ok=True)
    result_file_path = result_directory / Path(
        f"results_ENGINE_{engine_config_name}_NETWORK1_{network_name1}_NETWORK2_{network_name2}.txt"  # noqa: E501
    )
    store_experiment_params(
        namespace=args, result_file_path=result_file_path, source_file_path=__file__
    )

    # Read the weights and biases config file
    with open(WANDB_CONFIG_FILE, "r") as f:
        wandb_config = yaml.safe_load(f)

    if not DEBUG:
        run = wandb.init(config=wandb_config, project="rl-testing")
        wandb_config = wandb.config
    else:
        wandb_config = FakeConfig(DEBUG_CONFIG)

    # Initialize the result lists
    (
        populations,
        best_fitness_value_series,
        average_fitness_value_series,
        worst_fitness_value_series,
        unique_individual_fractions,
    ) = ([], [], [], [], [])

    # Run the evolutionary algorithm 'num_runs_per_config' times
    for seed in range(wandb_config.num_runs_per_config):
        print(f"Starting run {seed + 1}/{wandb_config.num_runs_per_config}")
        (
            population,
            best_fitness_values,
            average_fitness_values,
            worst_fitness_values,
            unique_individual_fraction,
        ) = evolutionary_algorithm(
            population_size=wandb_config.population_size,
            crossover_prob=wandb_config.crossover_prob,
            mutation_prob=wandb_config.mutation_prob,
            num_generations=wandb_config.num_generations,
            tournament_fraction=wandb_config.tournament_fraction,
            num_workers=wandb_config.num_workers,
            seed=seed,
        )
        populations.append(population)
        best_fitness_value_series.append(best_fitness_values)
        average_fitness_value_series.append(average_fitness_values)
        worst_fitness_value_series.append(worst_fitness_values)
        unique_individual_fractions.append(unique_individual_fraction)

    # Average the fitness values and unique individual fractions over all runs
    # and compute the standard deviation
    best_fitness_values = np.mean(best_fitness_value_series, axis=0)
    average_fitness_values = np.mean(average_fitness_value_series, axis=0)
    worst_fitness_values = np.mean(worst_fitness_value_series, axis=0)
    unique_individual_fraction = np.mean(unique_individual_fractions, axis=0)
    best_fitness_values_std = np.std(best_fitness_value_series, axis=0)
    average_fitness_values_std = np.std(average_fitness_value_series, axis=0)
    worst_fitness_values_std = np.std(worst_fitness_value_series, axis=0)
    unique_individual_fraction_std = np.std(unique_individual_fractions, axis=0)

    # Log the results
    if not DEBUG:
        for i in range(len(best_fitness_values)):
            wandb.log(
                {
                    "best_fitness_value_avg": best_fitness_values[i],
                    "best_fitness_value_std": best_fitness_values_std[i],
                    "average_fitness_value_avg": average_fitness_values[i],
                    "average_fitness_value_std": average_fitness_values_std[i],
                    "worst_fitness_value_avg": worst_fitness_values[i],
                    "worst_fitness_value_std": worst_fitness_values_std[i],
                    "unique_individual_fraction_avg": unique_individual_fraction[i],
                    "unique_individual_fraction_std": unique_individual_fraction_std[i],
                },
                step=i,
            )

    fitness_values = [individual.fitness for individual in population]
    best_individual = population[np.argmin(fitness_values)]
