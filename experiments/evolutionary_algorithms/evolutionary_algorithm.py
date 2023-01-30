import argparse
import logging
from pathlib import Path
from typing import List, Optional

import chess
import numpy as np
from deap import base, creator, tools

from rl_testing.config_parsers import get_engine_config
from rl_testing.engine_generators import get_engine_generator
from rl_testing.evolutionary_algorithms.crossovers import (
    Crossover,
    crossover_half_board,
    crossover_one_eighth_board,
    crossover_one_quarter_board,
)
from rl_testing.evolutionary_algorithms.fitness import PieceNumberFitness
from rl_testing.evolutionary_algorithms.individuals import BoardIndividual
from rl_testing.evolutionary_algorithms.mutations import (
    Mutator,
    mutate_add_one_piece,
    mutate_castling_rights,
    mutate_flip_board,
    mutate_move_one_piece_adjacent,
    mutate_player_to_move,
    mutate_remove_one_piece,
    mutate_rotate_board,
)
from rl_testing.evolutionary_algorithms.selections import Selector, select_tournament
from rl_testing.util.experiment import store_experiment_params

RESULT_DIR = Path(__file__).parent.parent / Path("results/evolutionary_algorithm")


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


def evolutionary_algorithm():
    random_state = np.random.default_rng(42)

    POPULATION_SIZE = 100
    CROSSOVER_PROB, MUTATION_PROB, N_GENERATIONS = 0.2, 0.5, 50

    # Initialize the fitness function
    fitness = PieceNumberFitness()

    # Initialize the mutation functions
    mutate = Mutator(
        mutation_strategy="all",
        _random_state=random_state,
    )
    mutate.register_mutation_function(
        [
            mutate_player_to_move,
            mutate_add_one_piece,
            mutate_remove_one_piece,
            mutate_flip_board,
            mutate_rotate_board,
            mutate_move_one_piece_adjacent,
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
        rounds=POPULATION_SIZE,
        tournament_size=POPULATION_SIZE // 3,
        is_bigger_better=fitness.is_bigger_better(),
    )

    # Create the population
    population = get_random_individuals("data/random_positions.txt", POPULATION_SIZE, random_state)

    # Evaluate the entire population
    fitness_values = map(fitness.evaluate, population)
    for individual, fitness_val in zip(population, fitness_values):
        individual.fitness = fitness_val

    for generation in range(N_GENERATIONS):
        print(f"{generation = }")
        # Select the next generation individuals
        offspring = select(population)

        # Clone the selected individuals
        offspring = list(map(chess.Board.copy, offspring))

        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random_state.random() < CROSSOVER_PROB:
                crossover(child1, child2)

        # Apply mutation on the offspring
        for mutant in offspring:
            if random_state.random() < MUTATION_PROB:
                mutate(mutant)

        # Evaluate the individuals with an invalid fitness
        unevaluated_individuals = [
            individual for individual in offspring if individual.fitness is None
        ]
        fitness_values = map(fitness.evaluate, unevaluated_individuals)
        for individual, fitness_val in zip(unevaluated_individuals, fitness_values):
            individual.fitness = fitness_val

        # The population is entirely replaced by the offspring
        population = offspring

        # Print the best individual and its fitness
        best_individual, best_fitness = fitness.best_individual(population)
        print(f"{best_individual = }, {best_fitness = }")

        # Print the worst individual and its fitness
        worst_individual, worst_fitness = fitness.worst_individual(population)
        print(f"{worst_individual = }, {worst_fitness = }")
        print()

    return population


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

    # Run the experiment
    population = evolutionary_algorithm()
    fitness_values = [individual.fitness for individual in population]
    best_individual = population[np.argmin(fitness_values)]
    print(str(best_individual))
