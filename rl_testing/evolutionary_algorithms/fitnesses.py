import abc
import asyncio
import logging
import os
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


import chess
import chess.engine
import numpy as np
from rl_testing.distributed.queue_utils import EmptySocketAddress, SocketAddress, build_manager

from rl_testing.engine_generators import EngineGenerator
from rl_testing.evolutionary_algorithms.individuals import BoardIndividual, Individual
from rl_testing.util.cache import LRUCache
from rl_testing.util.chess import cp2q, rotate_180_clockwise
from rl_testing.util.engine import RelaxedUciProtocol, engine_analyse
from rl_testing.util.util import get_task_result_handler
from rl_testing.distributed.worker import AnalysisObject
from rl_testing.distributed.distributed_queue_manager import (
    connect_to_manager,
)

from rl_testing.distributed.distributed_queue_manager import (
    QueueManager,
    default_address,
    default_port,
    default_password,
)

FEN = str


class Fitness(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            (hasattr(subclass, "use_async") and callable(subclass.use_async))
            and (hasattr(subclass, "best_individual") and callable(subclass.best_individual))
            and (hasattr(subclass, "worst_individual") and callable(subclass.worst_individual))
            and (hasattr(subclass, "is_bigger_better"))
            and (
                (hasattr(subclass, "evaluate") and callable(subclass.evaluate))
                or (hasattr(subclass, "evaluate_async") and callable(subclass.evaluate_async))
            )
            or NotImplemented
        )

    @property
    @abc.abstractmethod
    def use_async(self) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_bigger_better(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def best_individual(self, individuals: List[Individual]) -> Individual:
        raise NotImplementedError

    @abc.abstractmethod
    def worst_individual(self, individuals: List[Individual]) -> Individual:
        raise NotImplementedError

    def evaluate(self, individual: Individual) -> float:
        raise NotImplementedError

    async def evaluate_async(self, individuals: List[Individual]) -> List[float]:
        raise NotImplementedError

    @property
    def network_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _find_individual(self, individuals: List[Individual], direction: Callable) -> Individual:
        # Make sure that all individuals have a fitness value and compute it if not.
        for individual in individuals:
            if individual.fitness is None:
                raise ValueError(
                    "Individuals must have a fitness value before calling this method."
                )
                # individual.fitness = self.evaluate(individual)

        fitness_vals = np.array([individual.fitness for individual in individuals])
        return individuals[direction(fitness_vals)]


class PieceNumberFitness(Fitness):
    def __init__(self, more_pieces_better: bool = True) -> None:
        self._more_pieces_better = more_pieces_better

    @property
    def use_async(self) -> bool:
        return False

    @property
    def is_bigger_better(self) -> bool:
        return self._more_pieces_better

    @lru_cache(maxsize=200_000)
    def evaluate(self, board: BoardIndividual) -> float:
        num_pieces = float(len(board.piece_map()))
        return num_pieces if self._more_pieces_better else -num_pieces

    def best_individual(self, individuals: List[BoardIndividual]) -> BoardIndividual:
        return self._find_individual(individuals, np.argmax)

    def worst_individual(self, individuals: List[BoardIndividual]) -> BoardIndividual:
        return self._find_individual(individuals, np.argmin)


class EditDistanceFitness(Fitness):
    def __init__(self, target: str) -> None:
        self._target = self.prepare_fen(target)
        self.distance_cache: dict[Tuple[str, str], int] = {}
        self.max_cache_size = 100000

    def prepare_fen(self, fen: str) -> str:
        return " ".join(fen.split(" ")[:3])

    @property
    def use_async(self) -> bool:
        return False

    @property
    def is_bigger_better(self) -> bool:
        return False

    @lru_cache(maxsize=200_000)
    def evaluate(self, individual: BoardIndividual) -> float:
        if len(self.distance_cache) > self.max_cache_size:
            self.distance_cache: dict[Tuple[str, str], int] = {}
        return self.levenshtein_distance(self._target, self.prepare_fen(individual.fen()))

    def best_individual(self, individuals: List[BoardIndividual]) -> BoardIndividual:
        return self._find_individual(individuals, np.argmin)

    def worst_individual(self, individuals: List[BoardIndividual]) -> BoardIndividual:
        return self._find_individual(individuals, np.argmax)

    def levenshtein_distance(self, string1: str, string2):
        if (string1, string2) in self.distance_cache:
            return self.distance_cache[(string1, string2)]
        if len(string2) == 0:
            return len(string1)
        if len(string1) == 0:
            return len(string2)

        if string1[0] == string2[0]:
            return self.levenshtein_distance(string1[1:], string2[1:])

        distance = min(
            0.5 + self.levenshtein_distance(string1[1:], string2),
            0.5 + self.levenshtein_distance(string1, string2[1:]),
            1 + self.levenshtein_distance(string1[1:], string2[1:]),
        )

        self.distance_cache[(string1, string2)] = distance
        return distance


class BoardSimilarityFitness(Fitness):
    def __init__(self, target: str) -> None:
        self.piece_map = chess.Board(target).piece_map()

    @property
    def use_async(self) -> bool:
        return False

    @property
    def is_bigger_better(self) -> bool:
        return False

    def best_individual(self, individuals: List[Individual]) -> Individual:
        return self._find_individual(individuals, np.argmin)

    def worst_individual(self, individuals: List[Individual]) -> Individual:
        return self._find_individual(individuals, np.argmax)

    @lru_cache(maxsize=200_000)
    def evaluate(self, individual: BoardIndividual) -> float:
        fitness = 0.0
        test_piece_map = individual.piece_map()

        for square in chess.SQUARES:
            if square in self.piece_map and square not in test_piece_map:
                fitness += 0.5
            elif square not in self.piece_map and square in test_piece_map:
                fitness += 1.0
            elif square in self.piece_map and square in test_piece_map:
                if self.piece_map[square].color != test_piece_map[square].color:
                    fitness += 1.0
                elif self.piece_map[square].piece_type != test_piece_map[square].piece_type:
                    fitness += 0.5

        return fitness


class HashFitness(Fitness):
    @property
    def use_async(self) -> bool:
        return False

    @property
    def is_bigger_better(self) -> bool:
        return True

    def best_individual(self, individuals: List[Individual]) -> Individual:
        return self._find_individual(individuals, np.argmax)

    def worst_individual(self, individuals: List[Individual]) -> Individual:
        return self._find_individual(individuals, np.argmin)

    @lru_cache(maxsize=200_000)
    def evaluate(self, individual: BoardIndividual) -> float:
        assert hasattr(individual, "__hash__"), "Individual must have implemented a hash method"
        return hash(individual)


def less_pieces_fitness(individual: chess.Board) -> float:
    """A fitness function which rewards individuals with less pieces on the board.
    The fitness is normalized to lie in the interval [0, 0.5] where a higher value means a better fitness.

    Args:
        individual: The individual to evaluate.

    Returns:
        The fitness value of the individual.
    """
    # The minimum number of possible pieces is 2 (two kings)
    # The maximum number of possible pieces is 32 (16 white and 16 black pieces)
    return (1.0 - (len(individual.piece_map()) - 2) / 30.0) / 2


class BoardTransformationFitness(Fitness):
    @staticmethod
    async def write_output(
        input_queue: asyncio.Queue,
        result_file_path: str,
        identifier_str: str = "",
    ) -> None:
        buffer_limit = 1000
        with open(result_file_path, "r+") as result_file:
            buffer_size = 0
            current_results = result_file.read()

            if "fen1,fitness1,fen2,fitness2" not in current_results:
                result_file.write("fen1,fitness1,fen2,fitness2\n")
            while True:
                buffer_size += 1
                fen1, fitness1, fen2, fitness2 = await input_queue.get()
                result_file.write(f"{fen1},{fitness1},{fen2},{fitness2}\n")
                if buffer_size > 0 and buffer_size % buffer_limit == 0:
                    logging.info(f"[{identifier_str}] Write {buffer_limit} results to file")
                    result_file.flush()
                    os.fsync(result_file.fileno())
                    buffer_size = 0

                input_queue.task_done()

    def __init__(
        self,
        result_path: Optional[str] = None,
        address: str = default_address,
        port: int = default_port,
        password: str = default_password,
        required_engine_config_name: Optional[str] = None,
        network_state: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initializes the BoardTransformationFitness class.

        Args:
            result_path (Optional[str], optional): The path to the file where the results
                should be written. Defaults to None.
            address (str, optional): The address of the engine queues. Defaults to default_address.
            port (int, optional): The port of the engine queues. Defaults to default_port.
            password (str, optional): The password of the engine queues. Defaults to
                default_password.
            required_engine_config_name (Optional[str], optional): The name of the engine config
                to use. Defaults to None.
            network_state (Optional[Dict[str, Any]], optional): The network state to use.
                Defaults to None.
            logger (Optional[logging.Logger], optional): A logger to use. Defaults to None.
        """
        # Create a logger if it doesn't exist
        self.logger = logger or logging.getLogger(__name__)

        # Initialize all the variables
        self.result_path = result_path
        socket_address = SocketAddress(address=address, port=port, password=password)

        # Prepare the queues for the distributed fitness evaluation
        # If an existing network state is given, use it to initialize the engine
        if network_state is not None:
            self.net_manager = network_state["net_manager"]
        # Otherwise, create new network managers
        else:
            self.net_manager = build_manager(
                **socket_address.to_dict(),
                required_engine_config=required_engine_config_name,
            )
            self.net_manager.start()

        self.input_queue, self.output_queue = (
            self.net_manager.input_queue(),
            self.net_manager.output_queue(),
        )
        self.result_queue = asyncio.Queue()
        self.cache: Dict[FEN, float] = LRUCache(maxsize=200_000)

        self.result_task = None

        # Log how many times a position has been truly evaluated (not cached)
        self.num_evaluations = 0

    @property
    def network_state(self) -> Dict[str, Any]:
        return {
            "net_manager": self.net_manager,
        }

    async def create_tasks(self) -> None:
        handle_task_exception = get_task_result_handler(
            logger=self.logger, message="Task raised an exception"
        )

        # Create the task for writing the results to file
        if self.result_path is not None:
            self.result_task = asyncio.create_task(
                BoardTransformationFitness.write_output(
                    input_queue=self.result_queue,
                    result_file_path=self.result_path,
                    identifier_str="RESULT WRITER",
                )
            )

            self.result_task.add_done_callback(handle_task_exception)

    def cancel_tasks(self) -> None:
        """Cancels all the tasks."""

        if self.result_task is not None:
            self.result_task.cancel()

    @property
    def use_async(self) -> bool:
        return True

    @property
    def is_bigger_better(self) -> bool:
        return True

    def best_individual(self, individuals: List[Individual]) -> Individual:
        return self._find_individual(individuals, np.argmax)

    def worst_individual(self, individuals: List[Individual]) -> Individual:
        return self._find_individual(individuals, np.argmin)

    async def evaluate_async(self, individuals: List[BoardIndividual]) -> List[float]:
        """Evaluates the given individuals asynchronously.

        Args:
            individuals: The individuals to evaluate.

        Returns:
            The fitness values of the individuals.
        """
        # A dictionary to store fens which are currently being processed together with their
        # positions in the results list
        fens_in_progress: Set[FEN] = set()

        # Prepare the result list and fill it with a negative value. This fitness function only
        # produces positive values, so this is a good way to mark invalid individuals.
        results: List[float] = [-1.0] * len(individuals)
        result_fens: List[Tuple[FEN, FEN]] = []

        # An output dictionary to match the results of the two output queues
        output_dict: Dict[FEN, float] = {}

        print(f"Before processing: {len(individuals)})")

        # Iterate over the individuals and either compute their fitness or fetch the fitness
        # from the cache
        for index, individual in enumerate(individuals):
            fen1: FEN = individual.fen()
            fen2: FEN = chess.Board(fen1).transform(rotate_180_clockwise).fen()
            result_fens.append((fen1, fen2))

            for fen in [fen1, fen2]:
                if fen in self.cache:
                    output_dict[fen] = self.cache[fen]
                elif fen not in fens_in_progress:
                    fens_in_progress.add(fen)
                    self.input_queue.put(AnalysisObject(fen=fen))
                    self.num_evaluations += 1

        # Wait until all boards have been processed
        self.input_queue.join()

        # Extract all results from the first output queue
        while not self.output_queue.empty():
            analysis_object: AnalysisObject = self.output_queue.get()
            output_dict[analysis_object.fen] = analysis_object.score
            self.output_queue.task_done()

        # Extract all results from the second output queue and compute the score difference
        for index, (fen1, fen2) in enumerate(result_fens):
            # Both results are valid
            if output_dict[fen1] not in ["invalid", "nan", None] and output_dict[fen2] not in [
                "invalid",
                "nan",
                None,
            ]:
                fitness = abs(output_dict[fen1] - output_dict[fen2])

                results[index] = fitness

                # Add the scores to the cache
                self.cache[fen1] = output_dict[fen1]
                self.cache[fen2] = output_dict[fen2]

            # No matter the outcome, write the result to the result file
            if self.result_path is not None:
                await self.result_queue.put((fen1, self.cache[fen1], fen2, self.cache[fen2]))

        # Wait until all results have been written to file
        if self.result_path is not None:
            await self.result_queue.join()

        return results


class DifferentialTestingFitness(Fitness):
    @staticmethod
    async def write_output(
        input_queue: asyncio.Queue,
        result_file_path: str,
        identifier_str: str = "",
    ) -> None:
        buffer_limit = 1000
        with open(result_file_path, "r+") as result_file:
            buffer_size = 0
            current_results = result_file.read()

            if "fen,fitness" not in current_results:
                result_file.write("fen,fitness\n")
            while True:
                buffer_size += 1
                fen, fitness = await input_queue.get()
                result_file.write(f"{fen},{fitness}\n")
                if buffer_size > 0 and buffer_size % buffer_limit == 0:
                    logging.info(f"[{identifier_str}] Write {buffer_limit} results to file")
                    result_file.flush()
                    os.fsync(result_file.fileno())
                    buffer_size = 0

                input_queue.task_done()

    def __init__(
        self,
        result_path: Optional[str] = None,
        address1: str = default_address,
        port1: int = default_port,
        password1: str = default_password,
        address2: str = default_address,
        port2: int = default_port,
        password2: str = default_password,
        required_engine_config_name1: Optional[str] = None,
        required_engine_config_name2: Optional[str] = None,
        network_state: Optional[Dict[str, Any]] = None,
        reward_crashes: bool = False,
        reward_fewer_pieces: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initializes the DifferentialTestingFitness class.

        Args:
            result_path (Optional[str], optional): The path to the file where the results should be
                written. Defaults to None.
            address1 (str, optional): The address of the first pair of engine queues.
                Defaults to default_address.
            port1 (int, optional): The port of the first pair of engine queues.
                Defaults to default_port.
            password1 (str, optional): The password of the first pair of engine queues.
                Defaults to default_password.
            address2 (str, optional): The address of the second pair of engine queues.
                Defaults to default_address.
            port2 (int, optional): The port of the second pair of engine queues.
                Defaults to default_port.
            password2 (str, optional): The password of the second pair of engine queues.
                Defaults to default_password.
            required_engine_config_name1 (Optional[str], optional): The name of the first
                engine config. Defaults to None.
            required_engine_config_name2 (Optional[str], optional): The name of the second
                engine config. Defaults to None.
            network_state (Optional[Dict[str, Any]], optional): The network state to use.
                Defaults to None.
            reward_crashes (bool, optional): Whether to actively optimize for crashes.
                Defaults to False.
            reward_fewer_pieces (bool, optional): Whether to actively optimize for fewer pieces
                on the board. Defaults to False.
            logger (Optional[logging.Logger], optional): A logger to use. Defaults to None.
        """
        # Create a logger if it doesn't exist
        self.logger = logger or logging.getLogger(__name__)

        # Initialize all the variables
        self.result_path = result_path
        socket_address1 = SocketAddress(address=address1, port=port1, password=password1)
        socket_address2 = SocketAddress(address=address2, port=port2, password=password2)

        # If an existing network state is given, use it to initialize the engine
        if network_state is not None:
            self.net_manager1 = network_state["net_manager1"]
            self.net_manager2 = network_state["net_manager2"]

        # Otherwise, create new network managers
        else:
            # Prepare the queues for the distributed fitness evaluation
            self.net_manager1 = build_manager(
                **socket_address1.to_dict(),
                required_engine_config=required_engine_config_name1,
            )
            self.net_manager1.start()

            # Prepare the queues for the distributed fitness evaluation
            self.net_manager2 = build_manager(
                **socket_address2.to_dict(),
                required_engine_config=required_engine_config_name2,
            )
            self.net_manager2.start()

        self.input_queue1, self.output_queue1 = (
            self.net_manager1.input_queue(),
            self.net_manager1.output_queue(),
        )
        self.input_queue2, self.output_queue2 = (
            self.net_manager2.input_queue(),
            self.net_manager2.output_queue(),
        )

        self.result_queue = asyncio.Queue()
        self.cache: Dict[FEN, float] = LRUCache(maxsize=200_000)
        self.reward_crashes = reward_crashes
        self.reward_fewer_pieces = reward_fewer_pieces

        self.result_task = None

        # Log how many times a position has been truly evaluated (not cached)
        self.num_evaluations = 0

    @property
    def network_state(self) -> Dict[str, Any]:
        return {
            "net_manager1": self.net_manager1,
            "net_manager2": self.net_manager2,
        }

    async def create_tasks(self) -> None:
        handle_task_exception = get_task_result_handler(
            logger=self.logger, message="Task raised an exception"
        )

        # Create the task for writing the results to file
        if self.result_path is not None:
            self.result_task = asyncio.create_task(
                DifferentialTestingFitness.write_output(
                    input_queue=self.result_queue,
                    result_file_path=self.result_path,
                    identifier_str="RESULT WRITER",
                )
            )

            self.result_task.add_done_callback(handle_task_exception)

    def cancel_tasks(self) -> None:
        """Cancels all the tasks."""

        if self.result_task is not None:
            self.result_task.cancel()

    @property
    def use_async(self) -> bool:
        return True

    @property
    def is_bigger_better(self) -> bool:
        return True

    def best_individual(self, individuals: List[Individual]) -> Individual:
        return self._find_individual(individuals, np.argmax)

    def worst_individual(self, individuals: List[Individual]) -> Individual:
        return self._find_individual(individuals, np.argmin)

    async def evaluate_async(self, individuals: List[BoardIndividual]) -> List[float]:
        """Evaluates the given individuals asynchronously.

        Args:
            individuals: The individuals to evaluate.

        Returns:
            The fitness values of the individuals.
        """
        # A dictionary to store fens which are currently being processed together with
        # their positions in the results list
        fens_in_progress: Dict[FEN, List[int]] = {}

        # Prepare the result list and fill it with a negative value. This fitness function only
        # produces positive values, so this is a good way to mark invalid individuals.
        results: List[float] = [-1.0] * len(individuals)

        # Iterate over the individuals and either compute their fitness or fetch the fitness
        # from the cache
        for index, individual in enumerate(individuals):
            fen: FEN = individual.fen()
            if fen in self.cache:
                results[index] = self.cache[fen]
            elif fen not in fens_in_progress:
                fens_in_progress[fen] = [index]
                self.input_queue1.put(AnalysisObject(fen=fen))
                self.input_queue2.put(AnalysisObject(fen=fen))
                self.num_evaluations += 1
            else:
                fens_in_progress[fen].append(index)

        # Wait until all boards have been processed
        self.input_queue1.join()
        self.input_queue2.join()

        # An output dictionary to match the results of the two output queues
        output_dict: Dict[FEN, float] = {}

        # Extract all results from the first output queue
        while not self.output_queue1.empty():
            analysis_object: AnalysisObject = self.output_queue1.get()
            output_dict[analysis_object.fen] = analysis_object.score
            self.output_queue1.task_done()

        # Extract all results from the second output queue and compute the score difference
        while not self.output_queue2.empty():
            analysis_object: AnalysisObject = self.output_queue2.get()
            fen, score = analysis_object.fen, analysis_object.score

            # Both results are valid
            if (output_dict[fen] != "invalid" and score != "invalid") and (
                output_dict[fen] != "nan" and score != "nan"
            ):
                fitness = abs(output_dict[fen] - score)

            elif output_dict[fen] == "invalid" or score == "invalid":
                # Cache the invalid value anyway to prevent future re-computations
                # (and crashes) of the same board
                if self.reward_crashes:
                    fitness = 2.0
                else:
                    fitness = -1.0
            elif output_dict[fen] == "nan" or score == "nan":
                if self.reward_crashes:
                    fitness = 2.0
                else:
                    fitness = -2.0

            # Compute the fitness for having fewer pieces on the board
            board = chess.Board(fen)
            if self.reward_fewer_pieces:
                less_pieces_score = less_pieces_fitness(board)
            else:
                less_pieces_score = 0.0

            # Add the fitness value to all individuals with the same fen
            for index in fens_in_progress[fen]:
                results[index] = fitness + less_pieces_score

            # Add the fitness value to the cache
            self.cache[fen] = fitness + less_pieces_score

            # No matter the outcome, write the result to the result file
            if self.result_path is not None:
                await self.result_queue.put((fen, self.cache[fen]))

            self.output_queue2.task_done()

        # Wait until all results have been written to file
        if self.result_path is not None:
            await self.result_queue.join()

        return results


if __name__ == "__main__":
    # board1 = chess.Board("8/1p6/1p6/pPp1p1n1/P1P1P1k1/1K1P4/8/2B5 w - - 110 118")
    # board2 = chess.Board("r3qb1r/pppbk1p1/2np2np/4p2Q/2BPP3/2P5/PP3PPP/RNB2RK1 w - - 4 11")
    # fitness = PieceNumberFitness()

    # print("board1: ", fitness.evaluate(board1))
    # print("board2: ", fitness.evaluate(board2))
    pass
