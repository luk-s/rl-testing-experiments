import argparse
import asyncio
import logging
import multiprocessing
import time
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import chess.engine
import numpy as np
import wandb
import wandb.sdk
import yaml

from rl_testing.config_parsers import get_engine_config
from rl_testing.config_parsers.engine_config_parser import EngineConfig
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.evolutionary_algorithms.crossovers import (
    Crossover,
    crossover_half_board,
    crossover_one_eighth_board,
    crossover_one_quarter_board,
)
from rl_testing.evolutionary_algorithms.fitness import (
    BoardSimilarityFitness,
    DifferentialTestingFitness,
    EditDistanceFitness,
    Fitness,
    HashFitness,
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
    mutate_substitute_piece,
)
from rl_testing.evolutionary_algorithms.selections import (
    Selector,
    select_tournament_fast,
)
from rl_testing.util.chess import is_really_valid
from rl_testing.util.evolutionary_algorithm import should_decrease_probability
from rl_testing.util.experiment import (
    get_experiment_params_dict,
    store_experiment_params,
)

RESULT_DIR = Path(__file__).parent.parent / Path("results/evolutionary_algorithm")
WANDB_CONFIG_FILE = Path(__file__).parent.parent / Path(
    "configs/hyperparameter_tuning_configs/config_ea_edit_distance.yaml"
)
DEBUG = False
DEBUG_CONFIG = {
    "num_runs_per_config": 2,
    "num_workers": 8,
    "probability_decay": True,
    "early_stopping": True,
    "early_stopping_value": 1.9,
    "num_generations": 10,
    "population_size": 50,
    "mutation_prob": 0.6,
    "crossover_prob": 0.5,
    "tournament_fraction": 0.2,
    "mutate_add_one_piece": True,
    "mutate_castling_rights": True,
    "mutate_flip_board": True,
    "mutate_move_one_piece": True,
    "mutate_move_one_piece_adjacent": True,
    "mutate_move_one_piece_legal": True,
    "mutate_player_to_move": True,
    "mutate_remove_one_piece": True,
    "mutate_rotate_board": True,
    "mutate_substitute_piece": True,
    "crossover_half_board": True,
    "crossover_one_eighth_board": True,
    "crossover_one_quarter_board": True,
}

MUTATION_FUNCTIONS_DICT = {
    "mutate_add_one_piece": mutate_add_one_piece,
    "mutate_castling_rights": mutate_castling_rights,
    "mutate_flip_board": mutate_flip_board,
    "mutate_move_one_piece": mutate_move_one_piece,
    "mutate_move_one_piece_adjacent": mutate_move_one_piece_adjacent,
    "mutate_move_one_piece_legal": mutate_move_one_piece_legal,
    "mutate_player_to_move": mutate_player_to_move,
    "mutate_remove_one_piece": mutate_remove_one_piece,
    "mutate_rotate_board": mutate_rotate_board,
    "mutate_substitute_piece": mutate_substitute_piece,
}

CROSSOVER_FUNCTIONS_DICT = {
    "crossover_half_board": crossover_half_board,
    "crossover_one_eighth_board": crossover_one_eighth_board,
    "crossover_one_quarter_board": crossover_one_quarter_board,
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


class EvolutionaryAlgorithmConfig:
    def __init__(
        self,
        num_runs_per_config: int,
        num_workers: int,
        probability_decay: bool,
        early_stopping: bool,
        early_stopping_value: float,
        num_generations: int,
        population_size: int,
        mutation_prob: float,
        crossover_prob: float,
        tournament_fraction: float,
        mutation_functions: List[Callable],
        crossover_functions: List[Callable],
    ) -> None:
        self.num_runs_per_config = num_runs_per_config
        self.num_workers = num_workers
        self.probability_decay = probability_decay
        self.early_stopping = early_stopping
        self.early_stopping_value = early_stopping_value
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.tournament_fraction = tournament_fraction
        self.mutation_functions = mutation_functions
        self.crossover_functions = crossover_functions

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "EvolutionaryAlgorithmConfig":
        mutation_functions: List[Callable] = []
        for mutation_function in MUTATION_FUNCTIONS_DICT:
            if config[mutation_function]:
                mutation_functions.append(MUTATION_FUNCTIONS_DICT[mutation_function])

        crossover_functions: List[Callable] = []
        for crossover_function in CROSSOVER_FUNCTIONS_DICT:
            if config[crossover_function]:
                crossover_functions.append(CROSSOVER_FUNCTIONS_DICT[crossover_function])
        return cls(
            num_runs_per_config=config["num_runs_per_config"],
            num_workers=config["num_workers"],
            probability_decay=config["probability_decay"],
            early_stopping=config["early_stopping"],
            early_stopping_value=config["early_stopping_value"],
            num_generations=config["num_generations"],
            population_size=config["population_size"],
            mutation_prob=config["mutation_prob"],
            crossover_prob=config["crossover_prob"],
            tournament_fraction=config["tournament_fraction"],
            mutation_functions=mutation_functions,
            crossover_functions=crossover_functions,
        )

    @classmethod
    def from_wandb_config(
        cls, config: wandb.sdk.wandb_config.Config
    ) -> "EvolutionaryAlgorithmConfig":
        mutation_functions: List[Callable] = []
        for mutation_function in MUTATION_FUNCTIONS_DICT:
            if getattr(config, mutation_function):
                mutation_functions.append(MUTATION_FUNCTIONS_DICT[mutation_function])

        crossover_functions: List[Callable] = []
        for crossover_function in CROSSOVER_FUNCTIONS_DICT:
            if getattr(config, crossover_function):
                crossover_functions.append(CROSSOVER_FUNCTIONS_DICT[crossover_function])
        return cls(
            num_runs_per_config=config.num_runs_per_config,
            num_workers=config.num_workers,
            probability_decay=config.probability_decay,
            early_stopping=config.early_stopping,
            early_stopping_value=config.early_stopping_value,
            num_generations=config.num_generations,
            population_size=config.population_size,
            mutation_prob=config.mutation_prob,
            crossover_prob=config.crossover_prob,
            tournament_fraction=config.tournament_fraction,
            mutation_functions=mutation_functions,
            crossover_functions=crossover_functions,
        )


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

    individuals = []

    while len(individuals) < amount:
        # Randomly choose fen strings from the file
        fens = random_state.choice(lines, size=amount - len(individuals), replace=False)

        # Convert the fen strings to boards
        candidates = [BoardIndividual(fen) for fen in fens]

        # Add the valid boards to the list of individuals
        individuals.extend(
            [
                candidate
                for candidate in candidates
                if is_really_valid(candidate) and candidate not in individuals
            ]
        )

    return individuals


def setup_operators(
    random_state: np.random.Generator,
    is_bigger_fitness_better: bool,
    evolutionary_algorithm_config: EvolutionaryAlgorithmConfig,
) -> Tuple[Mutator, Crossover, Selector]:

    # Initialize the mutation functions
    mutate = Mutator(
        mutation_strategy="all",
        _random_state=random_state,
    )

    mutate_global_prob = 1 / len(evolutionary_algorithm_config.mutation_functions) * 2

    for mutation_function in evolutionary_algorithm_config.mutation_functions:
        mutate.register_mutation_function(
            [mutation_function],
            retries=5,
            check_game_not_over=True,
            clear_fitness_values=True,
        )

    mutate.set_global_probability(mutate_global_prob)

    # Initialize the crossover functions
    crossover = Crossover(
        crossover_strategy="all",
        _random_state=random_state,
    )

    crossover_global_prob = 1 / len(evolutionary_algorithm_config.crossover_functions) * 2

    for crossover_function in evolutionary_algorithm_config.crossover_functions:
        crossover.register_crossover_function(
            [crossover_function],
            retries=5,
            check_game_not_over=True,
            clear_fitness_values=True,
        )

    crossover.set_global_probability(crossover_global_prob)

    # Initialize the selection functions
    select = Selector(
        selection_strategy="all",
        _random_state=random_state,
    )
    select.register_selection_function(
        select_tournament_fast,
        tournament_size=int(
            evolutionary_algorithm_config.population_size
            * evolutionary_algorithm_config.tournament_fraction
        ),
        is_bigger_better=is_bigger_fitness_better,
    )

    return mutate, crossover, select


def log_time(start_time: float, message: str = ""):
    """Log the time since the start of the program.

    Args:
        start_time (float): The time the program started.
    """
    end_time = time.time()
    time_elapsed = end_time - start_time
    logging.info(f"Time {message}: {time_elapsed:.2f} seconds.")


async def evolutionary_algorithm(
    evolutionary_algorithm_config: EvolutionaryAlgorithmConfig,
    engine_generator1: EngineGenerator,
    engine_generator2: EngineGenerator,
    search_limits1: Dict[str, Any],
    search_limits2: Dict[str, Any],
    network_name1: Optional[str] = None,
    network_name2: Optional[str] = None,
    num_engines1: int = 1,
    num_engines2: int = 1,
    logger: Optional[logging.Logger] = None,
    seed: Optional[int] = None,
) -> Tuple[
    List[BoardIndividual],
    List[BestFitnessValue],
    List[AverageFitnessValue],
    List[WorstFitnessValue],
    List[UniqueIndividualFraction],
]:
    """Run a simple evolutionary algorithm with multiple mutation and crossover functions using differential testing on
    chess engines as fitness values. The function uses python asyncio to run multiple chess engines in separate processes
    locally or on a cluster. Furthermore, the function uses the multiprocessing module to run multiple workers in parallel
    to speed up the mutation and crossover functions.

    Args:
        engine_generator1 (EngineGenerator): A generator for the first chess engine.
        engine_generator2 (EngineGenerator): A generator for the second chess engine.
        search_limits1 (Dict[str, Any]): The search limits which the first engine uses for the analysis.
        search_limits2 (Dict[str, Any]): The search limits which the second engine uses for the analysis.
        network_name1 (Optional[str], optional): The name of the network which the first engine uses. Defaults to None.
        network_name2 (Optional[str], optional): The name of the network which the second engine uses. Defaults to None.
        num_engines1 (int, optional): How many engines of type 1 should be run in parallel to speed up the analysis.
        num_engines2 (int, optional): How many engines of type 2 should be run in parallel to speed up the analysis.
        logger (Optional[logging.Logger], optional): A logger which can be used to log the progress of the algorithm.
        num_workers (Optional[int], optional): How many workers should be used to run the mutations and crossovers in
            parallel. Defaults to None which means that all available cores are used.
        seed (Optional[int], optional): The seed for the random number generator. Defaults to None

    Returns:
        Tuple[
            List[BoardIndividual],
            List[BestFitnessValue],
            List[AverageFitnessValue],
            List[WorstFitnessValue],
            List[UniqueIndividualFraction],
        ]: The best individual of each generation, the best fitness value of each generation, the average fitness value of
            each generation, the worst fitness value of each generation and the fraction of unique individuals in each
            generation.
    """
    start_time = time.time()
    for parameter in [
        evolutionary_algorithm_config.crossover_prob,
        evolutionary_algorithm_config.mutation_prob,
        evolutionary_algorithm_config.tournament_fraction,
    ]:
        assert 0 <= parameter <= 1, f"Parameter {parameter = } must be between 0 and 1."

    if evolutionary_algorithm_config.num_workers is None:
        evolutionary_algorithm_config.num_workers = multiprocessing.cpu_count()

    random_state = np.random.default_rng(seed)

    # Initialize the operators
    fitness = DifferentialTestingFitness(
        engine_generator1=engine_generator1,
        engine_generator2=engine_generator2,
        search_limits1=search_limits1,
        search_limits2=search_limits2,
        network_name1=network_name1,
        network_name2=network_name2,
        num_engines1=num_engines1,
        num_engines2=num_engines2,
        logger=logger,
    )
    await fitness.create_tasks()

    mutate, crossover, select = setup_operators(
        random_state,
        is_bigger_fitness_better=fitness.is_bigger_better,
        evolutionary_algorithm_config=evolutionary_algorithm_config,
    )

    with multiprocessing.Pool(processes=evolutionary_algorithm_config.num_workers) as pool:
        best_individuals = []
        best_fitness_values = []
        average_fitness_values = []
        worst_fitness_values = []
        unique_individual_fractions = []
        chunk_size = (
            evolutionary_algorithm_config.population_size
            // evolutionary_algorithm_config.num_workers
            // 5
        )

        # Create the population
        log_time(start_time, "before creating population")
        population = get_random_individuals(
            "data/random_positions.txt",
            evolutionary_algorithm_config.population_size,
            random_state,
        )
        log_time(start_time, "after creating population")

        # Evaluate the entire population
        for individual, fitness_val in zip(population, await fitness.evaluate_async(population)):
            individual.fitness = fitness_val

        for generation in range(evolutionary_algorithm_config.num_generations):
            logging.info(f"\n\nGeneration {generation}")
            # Select the next generation individuals
            log_time(start_time, "before selecting")

            # prepare the chunk sizes for the selection
            chunk_sizes = [
                evolutionary_algorithm_config.population_size
                // evolutionary_algorithm_config.num_workers
            ] * evolutionary_algorithm_config.num_workers + [
                evolutionary_algorithm_config.population_size
                % evolutionary_algorithm_config.num_workers
            ]
            # Select the individuals in parallel
            offspring = []
            results_async = [
                pool.apply_async(
                    select,
                    # args=(population, chunk_size_i),
                    kwds={"individuals": population, "rounds": chunk_size_i},
                )
                for chunk_size_i in chunk_sizes
            ]
            for result_async in results_async:
                offspring += result_async.get()

            # Clone the selected individuals
            log_time(start_time, "before cloning")
            offspring: List[BoardIndividual] = list(map(BoardIndividual.copy, offspring))
            log_time(start_time, "after cloning")

            # Apply crossover on the offspring
            mated_children: List[BoardIndividual] = []
            couple_candidates = list(zip(offspring[::2], offspring[1::2]))
            random_values = random_state.random(size=len(offspring) // 2)

            # Filter out the individuals that will mate
            mating_candidates = [
                couple_candidates[i]
                for i, random_value in enumerate(random_values)
                if random_value < evolutionary_algorithm_config.crossover_prob
            ]
            single_children = [
                couple_candidates[i]
                for i, random_value in enumerate(random_values)
                if random_value >= evolutionary_algorithm_config.crossover_prob
            ]

            # Apply crossover on the mating candidates
            log_time(start_time, "before mating")
            for individual1, individual2 in chain(
                single_children, pool.imap(crossover, mating_candidates, chunksize=chunk_size)
            ):
                mated_children.append(individual1)
                mated_children.append(individual2)
            log_time(start_time, "after mating")

            # Apply mutation on the offspring
            mutated_children: List[BoardIndividual] = []
            random_values = random_state.random(size=len(mated_children))

            # Filter out the individuals that will mutate
            mutation_candidates = [
                mated_children[i]
                for i, random_value in enumerate(random_values)
                if random_value < evolutionary_algorithm_config.mutation_prob
            ]
            non_mutation_candidates = [
                mated_children[i]
                for i, random_value in enumerate(random_values)
                if random_value >= evolutionary_algorithm_config.mutation_prob
            ]

            # Apply mutation on the mutation candidates
            log_time(start_time, "before mutating")
            unevaluated_individuals: List[BoardIndividual] = []
            mutated_children: List[BoardIndividual] = []
            for individual in chain(
                pool.imap(mutate, mutation_candidates, chunksize=chunk_size),
                non_mutation_candidates,
            ):
                if individual.fitness is None:
                    unevaluated_individuals.append(individual)
                mutated_children.append(individual)
            log_time(start_time, "after mutating")

            # Evaluate the individuals with an invalid fitness
            log_time(start_time, "before evaluating")
            for individual, fitness_val in zip(
                unevaluated_individuals, await fitness.evaluate_async(unevaluated_individuals)
            ):
                individual.fitness = fitness_val
            log_time(start_time, "after evaluating")

            # The population is entirely replaced by the offspring
            population = mutated_children

            log_time(start_time, "before finding best individual")
            # Print the best individual and its fitness
            best_individual, best_fitness = fitness.best_individual(population)
            best_individuals.append(best_individual.copy())
            best_fitness_values.append(best_fitness)
            logging.info(f"{best_individual = }, {best_fitness = }")

            log_time(start_time, "before finding worst individual")
            # Print the worst individual and its fitness
            worst_individual, worst_fitness = fitness.worst_individual(population)
            worst_fitness_values.append(worst_fitness)
            logging.info(f"{worst_individual = }, {worst_fitness = }")

            log_time(start_time, "before finding average fitness")
            # Print the average fitness
            average_fitness = (
                sum(individual.fitness for individual in population)
                / evolutionary_algorithm_config.population_size
            )
            average_fitness_values.append(average_fitness)
            logging.info(f"{average_fitness = }")

            log_time(start_time, "before finding unique individuals")
            # Print the number of unique individuals
            unique_individuals = set([p.fen() for p in population])
            unique_individual_fractions.append(
                len(unique_individuals) / evolutionary_algorithm_config.population_size
            )
            logging.info(f"Number of unique individuals = {len(unique_individuals)}")

            # Print the adaption history of the best individual
            logging.info(f"{best_individual.history = }")

            # Check if the probabilities of mutation and crossover are too high
            if evolutionary_algorithm_config.probability_decay and should_decrease_probability(
                best_fitness, difference_threshold=0.5
            ):
                assert (
                    mutate.global_probability is not None
                ), "Global mutation probability must not be None"
                assert (
                    crossover.global_probability is not None
                ), "Global crossover probability must not be None"
                if mutate.global_probability > 1 / len(mutate.mutation_functions):
                    mutate.set_global_probability(mutate.global_probability / 2)
                    logging.info(f"{mutate.global_probability = }")
                if crossover.global_probability > 1 / len(crossover.crossover_functions):
                    crossover.set_global_probability(crossover.global_probability / 2)
                    logging.info(f"{crossover.global_probability = }")

            # Check if the best fitness is above the early stopping threshold
            if (
                evolutionary_algorithm_config.early_stopping
                and best_fitness >= evolutionary_algorithm_config.early_stopping_value
            ):
                logging.info("Early stopping!")
                break

    logging.info(f"Number of evaluations: {fitness.num_evaluations}")

    # Cancel all running subprocesses which the fitness evaluator spawned
    fitness.cancel_tasks()

    return (
        best_individuals,
        best_fitness_values,
        average_fitness_values,
        worst_fitness_values,
        unique_individual_fractions,
    )


async def run_evolutionary_algorithm_n_times(
    number_of_runs: int,
    evolutionary_algorithm_config: EvolutionaryAlgorithmConfig,
    engine_config1: EngineConfig,
    engine_config2: EngineConfig,
    command_line_args: argparse.Namespace,
    logger: logging.Logger,
) -> List[
    Tuple[
        List[BoardIndividual],
        List[BestFitnessValue],
        List[AverageFitnessValue],
        List[WorstFitnessValue],
        List[UniqueIndividualFraction],
    ]
]:
    """A wrapper function which first initializes the engine generators and then runs the evolutionary algorithm
    `number_of_runs` times. The results of each run are stored in a list and returned.

    Args:
        number_of_runs (int): The number of times the evolutionary algorithm should be run.
        evolutionary_algorithm_config (EvolutionaryAlgorithmConfig): A configuration object for the evolutionary algorithm.
        engine_config1 (EngineConfig): A configuration object for the first engine.
        engine_config2 (EngineConfig): A configuration object for the second engine.
        command_line_args (argparse.Namespace): The command line arguments containing further configuration parameters
            for the engines.
        logger (logging.Logger): A logger object.

    Returns:
        List[
            Tuple[
                List[BoardIndividual],
                List[BestFitnessValue],
                List[AverageFitnessValue],
                List[WorstFitnessValue],
                List[UniqueIndividualFraction],
            ]
        ]: A list of tuples containing one tuple for each run. Each tuple contains a list of the best individuals, the best
            fitness values, the average fitness values, the worst fitness values and the unique individual fractions for each
            generation.
    """

    # Create the engine generators. This must be done inside this function because the ssh connection is run using asyncio, and all
    # asyncio objects need to be created inside the same event loop.
    engine_generator1, engine_generator2 = [
        get_engine_generator(engine_config) for engine_config in [engine_config1, engine_config2]
    ]

    # If the two engine configs are the same, we can use the same engine generator
    if engine_config1 == engine_config2:
        engine_generator2 = engine_generator1

    # Run 'number_of_runs' many runs of the evolutionary algorithm
    result_tuples = []
    for seed_offset in range(number_of_runs):
        logger.info(f"\n\nStarting run {seed_offset + 1}/{number_of_runs}")
        result_tuples.append(
            await evolutionary_algorithm(
                evolutionary_algorithm_config=evolutionary_algorithm_config,
                engine_generator1=engine_generator1,
                engine_generator2=engine_generator2,
                search_limits1=engine_config1.search_limits,
                search_limits2=engine_config2.search_limits,
                network_name1=command_line_args.network_path1,
                network_name2=command_line_args.network_path2,
                num_engines1=command_line_args.num_engines1,
                num_engines2=command_line_args.num_engines2,
                logger=logger,
                seed=command_line_args.seed + seed_offset,
            )
        )

    # Transpose the list of tuples so that the different result lists are in the same tuple
    # This is the same as 2d matrix transposition
    result_tuples = list(map(list, zip(*result_tuples)))

    return result_tuples


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
    parser.add_argument("--seed",                type=int,  default=42)
    parser.add_argument("--engine_config_name1", type=str,  default="local_400_nodes.ini")  # noqa: E501
    parser.add_argument("--engine_config_name2", type=str,  default="local_400_nodes.ini")  # noqa: E501
    # parser.add_argument("--engine_config_name1", type=str,  default="remote_400_nodes.ini")  # noqa: E501
    # parser.add_argument("--engine_config_name2", type=str,  default="remote_400_nodes.ini")  # noqa: E501
    parser.add_argument("--network_path1",       type=str,  default="T807785-b124efddc27559564d6464ba3d213a8279b7bd35b1cbfcf9c842ae8053721207")  # noqa: E501
    parser.add_argument("--network_path2",       type=str,  default="T785469-600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2")  # noqa: E501
    parser.add_argument("--num_engines1" ,       type=int,  default=2)
    parser.add_argument("--num_engines2" ,       type=int,  default=2)
    parser.add_argument("--result_subdir",       type=str,  default="main_results")

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
    engine_config1, engine_config2 = [
        get_engine_config(
            config_name=name,
            config_folder_path=Path(__file__).parent.parent.absolute()
            / Path("configs/engine_configs/"),
        )
        for name in [args.engine_config_name1, args.engine_config_name2]
    ]

    # Create results-file-name
    engine_config_name1 = args.engine_config_name1[:-4]
    engine_config_name2 = args.engine_config_name2[:-4]
    network_name1 = args.network_path1.split("-")[0]
    network_name2 = args.network_path2.split("-")[0]

    if engine_config_name1 == engine_config_name2:
        engine_config_name = engine_config_name1
    else:
        engine_config_name = f"{engine_config_name1}_AND_{engine_config_name2}"

    # Store the experiment config in the results file
    result_directory = RESULT_DIR / args.result_subdir
    result_directory.mkdir(parents=True, exist_ok=True)
    result_file_path = result_directory / Path(
        f"results_ENGINES_{engine_config_name}_NETWORKS_{network_name1}_AND_{network_name2}.txt"  # noqa: E501
    )
    store_experiment_params(
        namespace=args, result_file_path=result_file_path, source_file_path=__file__
    )

    # Get the experiment config as a dictionary
    experiment_config = get_experiment_params_dict(namespace=args, source_file_path=__file__)

    # Log the experiment config
    experiment_config_str = "{\n"
    for key, value in experiment_config.items():
        experiment_config_str += f"    {key}: {value},\n"
    experiment_config_str += "\n}"

    logger.info(f"\nExperiment config:\n{experiment_config_str}")

    # Read the weights and biases config file
    with open(WANDB_CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    if not DEBUG:
        run = wandb.init(config=config, project="rl-testing")
        evolutionary_algorithm_config = EvolutionaryAlgorithmConfig.from_wandb_config(wandb.config)
    else:
        evolutionary_algorithm_config = EvolutionaryAlgorithmConfig.from_dict(DEBUG_CONFIG)

    # Run the evolutionary algorithm 'num_runs_per_config' times
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())

    (
        populations,
        best_fitness_value_series,
        average_fitness_value_series,
        worst_fitness_value_series,
        unique_individual_fractions,
    ) = asyncio.run(
        run_evolutionary_algorithm_n_times(
            number_of_runs=evolutionary_algorithm_config.num_runs_per_config,
            evolutionary_algorithm_config=evolutionary_algorithm_config,
            engine_config1=engine_config1,
            engine_config2=engine_config2,
            command_line_args=args,
            logger=logger,
        )
    )

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
