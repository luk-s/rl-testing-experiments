import abc
from typing import List, Literal, Optional, Union

import chess

from rl_testing.evolutionary_algorithms import (
    CrossoverName,
    MutationName,
    SelectionName,
)

AdaptionType = Union[MutationName, CrossoverName]


class Individual(metaclass=abc.ABCMeta):
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "fitness")
            and hasattr(subclass, "copy")
            and hasattr(subclass, "history")
        ) or NotImplemented

    @property
    @abc.abstractmethod
    def fitness(self) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def history(self) -> List[AdaptionType]:
        raise NotImplementedError

    @fitness.setter
    @abc.abstractmethod
    def fitness(self, value: float) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self) -> "Individual":
        raise NotImplementedError


class BoardIndividual(chess.Board, Individual):
    def __init__(self, *args, **kwargs) -> None:
        self._fitness: Optional[float] = None
        self._history: List[AdaptionType] = []
        super().__init__(*args, **kwargs)

    def get_fitness(self) -> bool:
        return self._fitness

    def set_fitness(self, value: float) -> None:
        self._fitness = value

    def del_fitness(self) -> None:
        self._fitness = None

    fitness = property(get_fitness, set_fitness, del_fitness, "Fitness of the individual.")

    def get_history(self) -> List[AdaptionType]:
        return self._history

    def set_history(self, value: List[AdaptionType]) -> None:
        self._history = value

    def del_history(self) -> None:
        self._history = []

    history = property(
        get_history,
        set_history,
        del_history,
        "History of adaptions that have been made to the individual.",
    )

    def copy(self, *args, **kwargs) -> "BoardIndividual":
        board = super().copy(*args, **kwargs)
        board._fitness = self._fitness
        board._history = list(self._history)
        return board

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoardIndividual):
            return NotImplemented

        return self.fen(en_passant="fen") == other.fen(en_passant="fen")

    def __hash__(self) -> int:
        return hash(self.fen(en_passant="fen"))


if __name__ == "__main__":
    board1 = BoardIndividual("8/1p6/1p6/pPp1p1n1/P1P1P1k1/1K1P4/8/2B5 w - - 110 118")
    board2 = BoardIndividual("r3qb1r/pppbk1p1/2np2np/4p2Q/2BPP3/2P5/PP3PPP/RNB2RK1 w - - 4 11")

    board1.test()
