from enum import Enum

from rl_testing.evolutionary_algorithms.crossovers import (
    CrossoverName,
    crossover_half_board,
    crossover_one_eighth_board,
    crossover_one_quarter_board,
)
from rl_testing.evolutionary_algorithms.mutations import (
    MutationName,
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
