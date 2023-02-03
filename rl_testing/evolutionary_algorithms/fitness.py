import abc
import asyncio
import logging
from typing import Any, Callable, Dict, List, Tuple

import chess
import chess.engine
import numpy as np

from rl_testing.evolutionary_algorithms.individuals import BoardIndividual, Individual
from rl_testing.util.engine import engine_analyse
from rl_testing.util.util import cp2q


class Fitness(metaclass=abc.ABCMeta):
    def __subclasshook__(cls, subclass):
        return (
            (hasattr(subclass, "use_async") and callable(subclass.use_async))
            and (hasattr(subclass, "best_individual") and callable(subclass.best_individual))
            and (hasattr(subclass, "worst_individual") and callable(subclass.best_individual))
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
    def best_individual(self, individuals: List[Individual]) -> Tuple[Individual, float]:
        raise NotImplementedError

    @abc.abstractmethod
    def worst_individual(self, individuals: List[Individual]) -> Tuple[Individual, float]:
        raise NotImplementedError

    def evaluate(self, individual: Individual) -> float:
        raise NotImplementedError

    async def evaluate_async(self, individual: Individual) -> float:
        raise NotImplementedError

    def _find_individual(
        self, individuals: List[Individual], direction: Callable
    ) -> Tuple[Individual, float]:
        # Make sure that all individuals have a fitness value and compute it if not.
        for individual in individuals:
            if individual.fitness is None:
                raise ValueError(
                    "Individuals must have a fitness value before calling this method."
                )
                # individual.fitness = self.evaluate(individual)

        fitness_vals = np.array([individual.fitness for individual in individuals])
        return individuals[direction(fitness_vals)], individuals[direction(fitness_vals)].fitness


class PieceNumberFitness(Fitness):
    def __init__(self, more_pieces_better: bool = True) -> None:
        self._more_pieces_better = more_pieces_better

    @property
    def use_async(self) -> bool:
        return False

    @property
    def is_bigger_better(self) -> bool:
        return self._more_pieces_better

    def evaluate(self, board: BoardIndividual) -> float:
        num_pieces = float(len(board.piece_map()))
        return num_pieces if self._more_pieces_better else -num_pieces

    def best_individual(self, individuals: List[BoardIndividual]) -> Tuple[BoardIndividual, float]:
        return self._find_individual(individuals, np.argmax)

    def worst_individual(
        self, individuals: List[BoardIndividual]
    ) -> Tuple[BoardIndividual, float]:
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

    def evaluate(self, individual: BoardIndividual) -> float:
        if len(self.distance_cache) > self.max_cache_size:
            self.distance_cache: dict[Tuple[str, str], int] = {}
        return self.levenshtein_distance(self._target, self.prepare_fen(individual.fen()))

    def best_individual(self, individuals: List[BoardIndividual]) -> Tuple[BoardIndividual, float]:
        return self._find_individual(individuals, np.argmin)

    def worst_individual(
        self, individuals: List[BoardIndividual]
    ) -> Tuple[BoardIndividual, float]:
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

    def best_individual(self, individuals: List[Individual]) -> Tuple[Individual, float]:
        return self._find_individual(individuals, np.argmin)

    def worst_individual(self, individuals: List[Individual]) -> Tuple[Individual, float]:
        return self._find_individual(individuals, np.argmax)

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

    def best_individual(self, individuals: List[Individual]) -> Tuple[Individual, float]:
        return self._find_individual(individuals, np.argmax)

    def worst_individual(self, individuals: List[Individual]) -> Tuple[Individual, float]:
        return self._find_individual(individuals, np.argmin)

    def evaluate(self, individual: BoardIndividual) -> float:
        assert hasattr(individual, "__hash__"), "Individual must have implemented a hash method"
        return hash(individual)


if __name__ == "__main__":
    board1 = chess.Board("8/1p6/1p6/pPp1p1n1/P1P1P1k1/1K1P4/8/2B5 w - - 110 118")
    board2 = chess.Board("r3qb1r/pppbk1p1/2np2np/4p2Q/2BPP3/2P5/PP3PPP/RNB2RK1 w - - 4 11")
    fitness = PieceNumberFitness()

    print("board1: ", fitness.evaluate(board1))
    print("board2: ", fitness.evaluate(board2))
