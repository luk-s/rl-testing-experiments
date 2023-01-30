import abc
from typing import Any, Callable, List, Tuple

import chess
import numpy as np

from rl_testing.evolutionary_algorithms.individuals import BoardIndividual, Individual


class Fitness(metaclass=abc.ABCMeta):
    def __subclasshook__(cls, subclass):
        return (
            (hasattr(subclass, "use_async") and callable(subclass.use_async))
            and (hasattr(subclass, "is_bigger_better") and callable(subclass.is_bigger_better))
            and (hasattr(subclass, "best_individual") and callable(subclass.best_individual))
            and (
                (hasattr(subclass, "evaluate") and callable(subclass.evaluate))
                or (hasattr(subclass, "evaluate_async") and callable(subclass.evaluate))
            )
            or NotImplemented
        )

    @property
    @abc.abstractmethod
    def use_async(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def is_bigger_better(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def best_individual(self, individuals: List[Individual]) -> Tuple[Individual, float]:
        raise NotImplementedError

    def evaluate(self, individual: Individual) -> float:
        raise NotImplementedError

    async def evaluate_async(self, individual: Individual) -> float:
        raise NotImplementedError


class PieceNumberFitness(Fitness):
    def use_async(self) -> bool:
        return False

    def is_bigger_better(self) -> bool:
        return False

    def evaluate(self, board: BoardIndividual) -> float:
        return float(len(board.piece_map()))

    def find_individual(
        self, individuals: List[BoardIndividual], direction: Callable
    ) -> Tuple[BoardIndividual, float]:
        # Make sure that all individuals have a fitness value and compute it if not.
        for individual in individuals:
            if individual.fitness is None:
                individual.fitness = self.evaluate(individual)

        fitness_vals = np.array([individual.fitness for individual in individuals])
        return individuals[direction(fitness_vals)], individuals[direction(fitness_vals)].fitness

    def best_individual(self, individuals: List[BoardIndividual]) -> Tuple[BoardIndividual, float]:
        return self.find_individual(individuals, np.argmax)

    def worst_individual(
        self, individuals: List[BoardIndividual]
    ) -> Tuple[BoardIndividual, float]:
        return self.find_individual(individuals, np.argmin)


if __name__ == "__main__":
    board1 = chess.Board("8/1p6/1p6/pPp1p1n1/P1P1P1k1/1K1P4/8/2B5 w - - 110 118")
    board2 = chess.Board("r3qb1r/pppbk1p1/2np2np/4p2Q/2BPP3/2P5/PP3PPP/RNB2RK1 w - - 4 11")
    fitness = PieceNumberFitness()

    print("board1: ", fitness.evaluate(board1))
    print("board2: ", fitness.evaluate(board2))
