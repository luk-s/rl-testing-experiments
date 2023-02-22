import argparse
import asyncio
import logging
import multiprocessing
import time
from itertools import chain
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import chess.engine
import numpy as np
import wandb.sdk
import yaml

import wandb
from experiments.evolutionary_algorithms.evolutionary_algorithm_configs import (
    SimpleEvolutionaryAlgorithmConfig,
)
from rl_testing.config_parsers import get_engine_config
from rl_testing.config_parsers.engine_config_parser import EngineConfig
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.evolutionary_algorithms import (
    get_initialized_crossover,
    get_initialized_mutator,
    get_initialized_selector,
)
from rl_testing.evolutionary_algorithms.algorithms import AsyncEvolutionaryAlgorithm
from rl_testing.evolutionary_algorithms.crossovers import Crossover
from rl_testing.evolutionary_algorithms.fitnesses import (
    BoardSimilarityFitness,
    DifferentialTestingFitness,
    EditDistanceFitness,
    Fitness,
    HashFitness,
    PieceNumberFitness,
)
from rl_testing.evolutionary_algorithms.individuals import BoardIndividual
from rl_testing.evolutionary_algorithms.mutations import Mutator
from rl_testing.evolutionary_algorithms.populations import Population, SimplePopulation
from rl_testing.evolutionary_algorithms.selections import (
    Selector,
    select_tournament_fast,
)
from rl_testing.evolutionary_algorithms.statistics import SimpleStatistics
from rl_testing.util.chess import is_really_valid
from rl_testing.util.evolutionary_algorithm import (
    get_random_individuals,
    should_decrease_probability,
)
from rl_testing.util.experiment import (
    get_experiment_params_dict,
    store_experiment_params,
)
from rl_testing.util.util import get_random_state, log_time

RESULT_DIR = Path(__file__).parent.parent / Path("results/evolutionary_algorithm")
CONFIG_FOLDER = Path(__file__).parent.parent
WANDB_CONFIG_FILE = CONFIG_FOLDER / Path(
    "configs/evolutionary_algorithm_configs/config_ea_differential_testing.yaml"
)
ENGINE_CONFIG_FOLDER = CONFIG_FOLDER / Path("configs/engine_configs")
EVOLUTIONARY_ALGORITHM_CONFIG_FOLDER = CONFIG_FOLDER / Path(
    "configs/evolutionary_algorithm_configs"
)
DEBUG = True
Time = float


class SimpleEvolutionaryAlgorithm(AsyncEvolutionaryAlgorithm):
    def __init__(
        self,
        evolutionary_algorithm_config: SimpleEvolutionaryAlgorithmConfig,
        experiment_config: Dict[str, Any],
        logger: logging.Logger,
    ):
        # Experiment configs
        self.experiment_config = experiment_config
        self.evolutionary_algorithm_config = evolutionary_algorithm_config
        self.logger = logger

        # Evolutionary algorithm configs
        self.num_generations = evolutionary_algorithm_config.num_generations
        self.crossover_probability = evolutionary_algorithm_config.crossover_probability
        self.mutation_probability = evolutionary_algorithm_config.mutation_probability
        self.early_stopping = evolutionary_algorithm_config.early_stopping
        self.early_stopping_value = evolutionary_algorithm_config.early_stopping_value
        self.probability_decay = evolutionary_algorithm_config.probability_decay

    async def initialize(self, seed: int) -> None:
        # Create the random state
        self.random_state = get_random_state(seed)

        # Create a multiprocessing pool
        self.pool = multiprocessing.Pool(processes=self.evolutionary_algorithm_config.num_workers)

        # Create the fitness function
        self.fitness = DifferentialTestingFitness(
            **self.experiment_config["fitness_config"],
            logger=self.logger,
        )

        # Create the evolutionary operators
        self.mutate: Mutator = get_initialized_mutator(self.evolutionary_algorithm_config)
        self.crossover: Crossover = get_initialized_crossover(self.evolutionary_algorithm_config)
        self.select: Selector = get_initialized_selector(self.evolutionary_algorithm_config)

        # Create the population
        individuals = get_random_individuals(
            "data/random_positions.txt",
            self.evolutionary_algorithm_config.population_size,
            self.random_state,
        )
        self.population = SimplePopulation(
            individuals=individuals,
            fitness=self.fitness,
            mutator=self.mutate,
            crossover=self.crossover,
            selector=self.select,
            pool=self.pool,
            _random_state=self.random_state,
        )

        await self.fitness.create_tasks()

    async def run(self) -> SimpleStatistics:
        start_time = time.time()

        # Create the statistics
        statistics = SimpleStatistics()

        # Evaluate the entire population
        await self.population.evaluate_individuals_async()

        for generation in range(self.num_generations):
            logging.info(f"\n\nGeneration {generation}")

            # Select the next generation individuals
            log_time(start_time, "before selecting")
            self.population.create_next_generation()

            # Apply crossover on the offspring
            log_time(start_time, "before mating")
            self.population.crossover_individuals(self.crossover_probability)

            # Apply mutation on the offspring
            log_time(start_time, "before mutating")
            self.population.mutate_individuals(self.mutation_probability)

            # Evaluate the individuals with an invalid fitness
            log_time(start_time, "before evaluating")
            await self.population.evaluate_individuals_async()

            log_time(start_time, "before updating the statistics")
            statistics.update_time_series(self.population, log_statistics=True)

            # Check if the best fitness is above the early stopping threshold
            if (
                self.early_stopping
                and statistics.best_fitness_values[-1] >= self.early_stopping_value
            ):
                statistics.fill_time_series(generation, self.num_generations, self.population)
                logging.info("Early stopping!")
                break

            # Check if the probabilities of mutation and crossover are too high
            if self.probability_decay and should_decrease_probability(
                statistics.best_fitness_values[-1], difference_threshold=0.5
            ):
                logging.info("Decreasing mutation and crossover probabilities")
                self.mutate.multiply_probabilities(factor=0.5, log=True)
                self.crossover.multiply_probabilities(factor=0.5, log=True)

        end_time = time.time()
        logging.info(f"Number of evaluations: {self.fitness.num_evaluations}")
        logging.info(f"Total time: {end_time - start_time} seconds")
        statistics.set_scalars(
            runtime=end_time - start_time, num_evaluations=self.fitness.num_evaluations
        )

        # Log the best individual and its fitness
        best_individual = self.fitness.best_individual(statistics.best_individuals)
        best_fitness = best_individual.fitness
        logging.info(
            f"FINAL best individual: {best_individual.fen()}, FINAL best fitness: {best_fitness}"
        )

        # Log the histories of the mutation- and crossover probability distributions
        self.mutate.print_mutation_probability_history()
        self.crossover.print_crossover_probability_history()

        return statistics

    async def cleanup(self) -> None:
        # Cancel all running subprocesses which the fitness evaluator spawned
        self.fitness.cancel_tasks()

        self.pool.close()

        del self.fitness
        del self.mutate
        del self.crossover
        del self.select
        del self.population


async def main(experiment_config_dict: Dict[str, Any], logger: logging.Logger) -> None:
    """Run the experiment."""

    # Extract path to evolutionary algorithm config file
    evolutionary_algorithm_config_name = experiment_config_dict[
        "evolutionary_algorithm_config_name"
    ]

    # Build evolutionary algorithm config
    evolutionary_algorithm_config = SimpleEvolutionaryAlgorithmConfig.from_yaml_file(
        EVOLUTIONARY_ALGORITHM_CONFIG_FOLDER / evolutionary_algorithm_config_name
    )

    # Create the evolutionary algorithm
    evolutionary_algorithm = SimpleEvolutionaryAlgorithm(
        evolutionary_algorithm_config=evolutionary_algorithm_config,
        experiment_config=experiment_config_dict,
        logger=logger,
    )

    # Repeat multiple times
    run_statistics = []
    start_seed = experiment_config_dict["seed"]
    for run_id in range(evolutionary_algorithm_config.num_runs_per_config):
        # Initialize evolutionary algorithm object
        await evolutionary_algorithm.initialize(start_seed + run_id)

        # Run evolutionary algorithm
        run_statistics.append(await evolutionary_algorithm.run())

        # Cleanup evolutionary algorithm object
        await evolutionary_algorithm.cleanup()

    return run_statistics


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
    # parser.add_argument("--evolutionary_algorithm_config_name", type=str,  default="config_simple_population.yaml")  # noqa: E501
    parser.add_argument("--evolutionary_algorithm_config_name", type=str,  default="config_simple_population_small.yaml")  # noqa: E501
    # parser.add_argument("--engine_config_name1", type=str,  default="local_400_nodes.ini")  # noqa: E501
    # parser.add_argument("--engine_config_name2", type=str,  default="local_400_nodes.ini")  # noqa: E501
    parser.add_argument("--engine_config_name1", type=str,  default="remote_400_nodes.ini")  # noqa: E501
    parser.add_argument("--engine_config_name2", type=str,  default="remote_400_nodes.ini")  # noqa: E501
    parser.add_argument("--network_name1",       type=str,  default="T807785-b124efddc27559564d6464ba3d213a8279b7bd35b1cbfcf9c842ae8053721207")  # noqa: E501
    parser.add_argument("--network_name2",       type=str,  default="T785469-600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2")  # noqa: E501
    parser.add_argument("--num_engines1" ,       type=int,  default=2)
    parser.add_argument("--num_engines2" ,       type=int,  default=2)
    parser.add_argument("--result_subdir",       type=str,  default="main_results")
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

    # Get the experiment config as a dictionary
    experiment_config_dict = get_experiment_params_dict(namespace=args, source_file_path=__file__)

    np.random.seed(args.seed)

    # Get the engine configs
    engine_config1, engine_config2 = [
        get_engine_config(
            config_name=name,
            config_folder_path=ENGINE_CONFIG_FOLDER,
        )
        for name in [
            experiment_config_dict["engine_config_name1"],
            experiment_config_dict["engine_config_name2"],
        ]
    ]

    # Build the engine generators
    engine_generator1, engine_generator2 = [
        get_engine_generator(engine_config) for engine_config in [engine_config1, engine_config2]
    ]

    if engine_config1 == engine_config2:
        engine_generator2 = engine_generator1

    # Build the configs for the fitness function
    experiment_config_dict["fitness_config"] = {}
    for parameter in [
        "network_name1",
        "network_name2",
        "num_engines1",
        "num_engines2",
    ]:
        experiment_config_dict["fitness_config"][parameter] = experiment_config_dict[parameter]
        del experiment_config_dict[parameter]

    experiment_config_dict["fitness_config"]["engine_generator1"] = engine_generator1
    experiment_config_dict["fitness_config"]["engine_generator2"] = engine_generator2
    experiment_config_dict["fitness_config"]["search_limits1"] = engine_config1.search_limits
    experiment_config_dict["fitness_config"]["search_limits2"] = engine_config2.search_limits

    # Log the experiment config
    experiment_config_str = "{\n"
    for key, value in experiment_config_dict.items():
        experiment_config_str += f"    {key}: {value},\n"
    experiment_config_str += "\n}"

    logger.info(f"\nExperiment config:\n{experiment_config_str}")

    # Run the evolutionary algorithm 'num_runs_per_config' times
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())

    statistics: List[SimpleStatistics] = asyncio.run(main(experiment_config_dict, logger))

    # Average the fitness values and unique individual fractions over all runs
    # and compute the standard deviation
    averaged_statistics = SimpleStatistics.average_statistics(statistics)

    # Log the results
    averaged_statistics.log_statistics(use_wandb=not DEBUG)
