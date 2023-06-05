import chess
import numpy as np
import logging
from typing import Optional, Tuple, Dict, List
from rl_testing.util.chess import (
    is_really_valid,
)
from rl_testing.util.util import get_random_state
import itertools
import random


def _crossover_exchange_piece_pairs_build_candidates(
    board1: chess.Board,
    board2: chess.Board,
    piece_combination1: Tuple[int, int],
    piece_combination2: Tuple[int, int],
) -> Tuple[chess.Board, chess.Board]:
    board1_original = board1.copy()
    board2_original = board2.copy()

    # Add the new pieces
    board1_new_pos1, board1_new_pos2 = piece_combination2
    board1_new_piece1, board1_new_piece2 = board2.piece_at(piece_combination2[0]), board2.piece_at(
        piece_combination2[1]
    )
    board2_new_pos1, board2_new_pos2 = piece_combination1
    board2_new_piece1, board2_new_piece2 = board1.piece_at(piece_combination1[0]), board1.piece_at(
        piece_combination1[1]
    )
    board1.set_piece_at(board1_new_pos1, board1_new_piece1)
    board1.set_piece_at(board1_new_pos2, board1_new_piece2)
    board2.set_piece_at(board2_new_pos1, board2_new_piece1)
    board2.set_piece_at(board2_new_pos2, board2_new_piece2)

    # Remove the old pieces
    if piece_combination1[0] not in piece_combination2:
        board1.remove_piece_at(piece_combination1[0])
    if piece_combination1[1] not in piece_combination2:
        board1.remove_piece_at(piece_combination1[1])
    if piece_combination2[0] not in piece_combination1:
        board2.remove_piece_at(piece_combination2[0])
    if piece_combination2[1] not in piece_combination1:
        board2.remove_piece_at(piece_combination2[1])

    # Ensure that the number of pieces is 8
    # Ensure that the number of pieces is 8
    if len(board1.piece_map()) != 8:
        print(f"Original board1: {board1_original.fen()}")
        print(f"Original board2: {board2_original.fen()}")
        print(
            f"Square combination1: {[chess.square_name(square) for square in piece_combination1]}"
        )
        print(
            f"Square combination2: {[chess.square_name(square) for square in piece_combination2]}"
        )
        raise ValueError(f"Board1 has {len(board1.piece_map())} pieces.")

    if len(board2.piece_map()) != 8:
        print(f"Original board1: {board1_original.fen()}")
        print(f"Original board2: {board2_original.fen()}")
        print(
            f"Square combination1: {[chess.square_name(square) for square in piece_combination1]}"
        )
        print(
            f"Square combination2: {[chess.square_name(square) for square in piece_combination2]}"
        )
        raise ValueError(f"Board2 has {len(board2.piece_map())} pieces.")

    return board1, board2


def crossover_exchange_piece_pairs(
    board1: chess.Board,
    board2: chess.Board,
    ensure_single_kings: bool = True,
    _random_state: Optional[np.random.Generator] = None,
) -> Tuple[chess.Board, chess.Board]:
    """Crossover function that swaps one pair of pieces of the same type and opposite color with each other.
    E.G. board1 could have a white and black Knight, and board2 could have a white and black Bishop. This function
    would then move the two Knights on board2 and the two Bishops on board1.

    This function expects that the boards have symmetric positions, where White and Black have the same pieces.

    Args:
        board1 (chess.Board): First board.
        board2 (chess.Board): Second board.
        ensure_single_kings (bool, optional): Whether to ensure that the boards have only one king per color after the crossover.
        _random_state (Optional[np.random.Generator], optional): The random state to use. Defaults to None.

    Returns:
        Tuple[chess.Board, chess.Board]: The two boards after the crossover.
    """
    random_state = get_random_state(_random_state)

    # Get the two piece maps
    piece_map1 = board1.piece_map()
    piece_map2 = board2.piece_map()

    def create_piece_combinations(
        piece_map: Dict[chess.Square, chess.Piece]
    ) -> List[Tuple[chess.Square, chess.Square]]:
        # Separate white and black pieces
        white_pieces = {
            square: piece for square, piece in piece_map.items() if piece.color == chess.WHITE
        }
        black_pieces = {
            square: piece for square, piece in piece_map.items() if piece.color == chess.BLACK
        }

        # Create a list of piece-combinations of pieces of the same type and opposite color
        piece_combinations = []
        for piece_type in [chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING]:
            white_piece_squares = [
                square for square, piece in white_pieces.items() if piece.piece_type == piece_type
            ]
            black_piece_squares = [
                square for square, piece in black_pieces.items() if piece.piece_type == piece_type
            ]

            # Create all possible combinations of pieces of the same type and opposite color
            piece_combinations.extend(
                list(itertools.product(white_piece_squares, black_piece_squares))
            )

        return piece_combinations

    # Create all possible piece combinations for the two boards
    piece_combinations1 = create_piece_combinations(piece_map1)
    piece_combinations2 = create_piece_combinations(piece_map2)

    # Create all possible combinations of piece combinations
    final_piece_combinations = list(itertools.product(piece_combinations1, piece_combinations2))

    # Check for each combination whether this is a valid crossover, i.e. whether there wouldn't be
    # too many pieces of the same type and color on one board
    valid_piece_combinations = []
    for piece_combination1, piece_combination2 in final_piece_combinations:
        # Check that the squares would not remove some other piece from the board
        combination_invalid = False
        for square in piece_combination1:
            if square not in piece_combination2 and board2.piece_at(square) is not None:
                combination_invalid = True
                break

        for square in piece_combination2:
            if square not in piece_combination1 and board1.piece_at(square) is not None:
                combination_invalid = True
                break

        if combination_invalid:
            continue

        # Get the pieces on the boards
        piece1_1 = piece_map1[piece_combination1[0]]
        piece2_1 = piece_map1[piece_combination1[1]]
        piece1_2 = piece_map2[piece_combination2[0]]
        piece2_2 = piece_map2[piece_combination2[1]]

        # Assert that the pieces coming from the same board are of the same type
        assert (
            piece1_1.piece_type == piece2_1.piece_type
        ), f"Piece types of {piece1_1} and {piece2_1} are not the same."
        assert (
            piece1_2.piece_type == piece2_2.piece_type
        ), f"Piece types of {piece1_2} and {piece2_2} are not the same."

        # Extract the piece types of the two boards
        piece_type1 = piece1_1.piece_type
        piece_type2 = piece1_2.piece_type

        # Get the number of pieces of piece_type2 which currently are on board1
        num_pieces1 = sum(
            [
                piece.piece_type == piece_type2 and piece.color == chess.WHITE
                for piece in piece_map1.values()
            ]
        )

        # Get the number of pieces of piece_type1 which currently are on board2
        num_pieces2 = sum(
            [
                piece.piece_type == piece_type1 and piece.color == chess.WHITE
                for piece in piece_map2.values()
            ]
        )
        max_pieces_allowed = {
            chess.ROOK: 2,
            chess.KNIGHT: 2,
            chess.BISHOP: 2,
            chess.QUEEN: 1,
            chess.KING: 1,
        }

        # Filter out the combination if it would lead to too many pieces of the same type on
        # one board
        if (num_pieces1 + 1 > max_pieces_allowed[piece_type1] and piece_type1 != piece_type2) or (
            num_pieces2 + 1 > max_pieces_allowed[piece_type2] and piece_type1 != piece_type2
        ):
            continue

        # Build the new boards and check if they are valid
        new_board1, new_board2 = _crossover_exchange_piece_pairs_build_candidates(
            board1.copy(), board2.copy(), piece_combination1, piece_combination2
        )

        # Check if the new boards are valid
        if not is_really_valid(new_board1) or not is_really_valid(new_board2):
            continue

        # If the combination is valid, add it to the list of valid combinations
        valid_piece_combinations.append((piece_combination1, piece_combination2))

    # If there are no valid piece combinations, return the original boards
    if len(valid_piece_combinations) == 0:
        return board1, board2

    # Select a random piece combination
    piece_combination1, piece_combination2 = random_state.choice(valid_piece_combinations)

    # Reverse the piece combinations with a probability of 0.5
    if random_state.choice([True, False]):
        piece_combination1 = piece_combination1[::-1]

    # Build the new boards
    board1, board2 = _crossover_exchange_piece_pairs_build_candidates(
        board1, board2, piece_combination1, piece_combination2
    )

    return board1, board2


if __name__ == "__main__":
    fens = [
        "8/1B2N1n1/3k4/3N4/b7/8/n7/6K1 w - - 2 63",
        "2Bb4/4q3/8/B7/2b5/5K2/3Q4/1k6 w - - 13 47",
        "8/b5B1/3k4/1r6/8/3N4/n7/5K1R w - - 14 68",
        "K5n1/4q3/2b1k3/8/3QN3/8/1B6/8 w - - 4 44",
        "1R6/2K5/n1B5/1N6/k7/8/r7/4b3 w - - 8 58",
        "R7/1N2B3/8/8/6br/8/K5n1/5k2 w - - 8 59",
        "1B6/5b2/k7/6K1/2n3N1/3n4/7N/8 w - - 4 33",
        "8/2bk4/K1n5/4n3/3N4/2B5/N7/8 w - - 0 70",
        "7k/4R3/2r1b3/8/5R2/8/2r5/1B3K2 w - - 13 42",
        "5N2/8/2n5/2N1r3/1K6/5n1k/1R6/8 w - - 3 35",
        "5b2/8/k2q1rQ1/8/8/K7/2B5/1R6 w - - 12 48",
        "1N1n4/8/8/2b5/8/1B4B1/8/bk2K3 w - - 3 35",
        "r4K2/8/6q1/3b4/7Q/2B1R3/8/6k1 w - - 3 50",
        "8/8/8/3k4/7r/2B1R3/K3R2r/3b4 w - - 15 65",
        "2rr3k/8/8/8/B7/2b5/2R4K/5R2 w - - 14 49",
        "1n1bB2N/8/2K5/8/8/2k5/2n4N/8 w - - 11 65",
        "8/8/5k1b/8/1B6/2R1r3/4nN2/2K5 w - - 3 45",
        "5K2/6R1/6qR/6r1/8/8/k7/6rQ w - - 1 51",
        "1B6/8/8/8/1k6/6B1/1b1n4/bK4N1 w - - 9 62",
        "8/n4N2/7N/3k4/1K6/5n2/6R1/5r2 w - - 6 30",
        "4k3/6q1/R7/5b2/4r3/Q7/7B/7K w - - 2 42",
        "3kN3/5n1N/8/3n4/8/2r5/4R3/6K1 w - - 0 48",
        "8/1k6/8/8/1K6/n6N/Bb4N1/5n2 w - - 12 60",
        "1R6/3bN3/8/7B/4k3/r4n2/6K1/8 w - - 11 40",
        "K1N2b2/B4r2/4R3/8/k7/8/6n1/8 w - - 3 51",
        "1K1n4/r5R1/k7/r7/8/6R1/7N/8 w - - 3 52",
        "rb6/8/3k4/2R5/8/1b6/5B2/BK6 w - - 16 58",
        "7b/3R4/8/5kr1/6q1/1B2Q2K/8/8 w - - 10 48",
        "1n5N/8/8/3B4/4R3/3b1kr1/8/3K4 w - - 2 32",
        "7N/3RK1r1/5R2/8/8/8/n1r5/1k6 w - - 0 33",
        "8/1b6/3K2kr/3R4/5r2/8/6B1/6R1 w - - 9 32",
        "N2B4/n7/8/1K6/6n1/2N5/2b5/4k3 w - - 4 49",
        "2k3n1/8/3R4/6r1/8/6NK/8/1n3N2 w - - 13 62",
        "n3N3/4b3/8/1Q6/2B4q/8/8/2K1k3 w - - 6 31",
        "7K/8/5r2/R4Rb1/2B4k/8/6r1/8 w - - 12 63",
        "7Q/5N2/8/8/8/n3NK2/3kn3/2q5 w - - 11 47",
        "6R1/7r/3k4/5b2/6R1/1B6/2K5/4r3 w - - 8 49",
        "5K2/2r5/3R4/8/7Q/8/1nq5/2k2N2 w - - 16 67",
        "6R1/1r6/1n6/7k/5K2/8/1B3b2/2N5 w - - 18 57",
        "1K6/8/2n5/2B5/N7/7n/3N3k/6b1 w - - 2 56",
        "3kq3/8/1R6/2r3b1/2B5/2Q5/8/2K5 w - - 5 64",
        "8/2B5/3K3N/1k6/8/1b4n1/2b3B1/8 w - - 10 36",
        "8/3n1N2/8/7K/1k6/8/2r4b/B4R2 w - - 6 45",
        "6b1/4Qn2/8/8/1N3q2/8/4K3/2kB4 w - - 4 47",
        "1R6/n3N3/6B1/r4b2/5k2/8/8/6K1 w - - 1 66",
        "5b1k/1B6/8/4r3/3KB3/8/7b/R7 w - - 2 60",
        "B7/b7/6n1/8/1K4N1/4r3/7R/5k2 w - - 7 58",
        "3R4/7R/3B4/r4r2/8/b7/2k3K1/8 w - - 4 70",
        "N4b2/7b/8/K7/B7/2kn4/7B/8 w - - 13 42",
        "8/8/r7/2bk1B2/2Nn1R2/1K6/8/8 w - - 0 51",
        "1Q5R/8/r7/8/2q2K2/2r3R1/8/k7 w - - 3 62",
        "8/8/5q2/8/K7/2NQ4/1R2nk1r/8 w - - 8 65",
        "1b6/4R3/4K1B1/1k6/8/6n1/7r/1N6 w - - 0 51",
        "6Q1/3R2K1/8/7N/2n5/2k5/8/2r3q1 w - - 1 57",
        "8/5n2/N7/8/1Nn3K1/k7/b7/3B4 w - - 4 35",
        "8/R7/8/2Nk4/K3n3/3N4/8/1r1n4 w - - 8 53",
        "1b2n3/8/8/1N1B3K/8/8/3k4/4b1B1 w - - 9 45",
        "8/8/8/4R3/1K4r1/1N6/6r1/2R3nk w - - 11 38",
        "N7/2K2B2/8/4k3/8/b4Q2/8/q1n5 w - - 15 67",
        "8/2N5/7k/8/5n2/1r1q4/R1Q5/K7 w - - 3 41",
    ]

    random_generator = np.random.default_rng(42)
    random.seed(42)

    valid_counter = 0

    for _ in range(1000):
        fen1 = "b7/8/K1k5/8/4R3/1q1r3B/6Q1/8 w - - 3 52"
        fen2 = "8/1Q6/r5q1/3R4/8/5k2/8/5K1n w - - 0 52"
        board1 = chess.Board(fen1)
        board2 = chess.Board(fen2)
        new_board1, new_board2 = crossover_exchange_piece_pairs(
            board1, board2, _random_state=random_generator
        )
        print("=" * 50)
        print(fen1)
        print(fen2)
        print(new_board1.fen())
        print(new_board2.fen())
        print("Valid new boards? ", is_really_valid(new_board1), is_really_valid(new_board2))
        valid_counter += int(is_really_valid(new_board1) and is_really_valid(new_board2))

    print(f"{valid_counter}/100 valid new board tuples.")
