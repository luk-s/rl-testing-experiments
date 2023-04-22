from pathlib import Path

from rl_testing.config_parsers import get_data_generator_config
from rl_testing.data_generators import get_data_generator

if __name__ == "__main__":
    DATA_CONFIG_NAME = "forced_moves_fen.ini"
    NUM_POSITIONS = 1_000_000

    data_config = get_data_generator_config(
        DATA_CONFIG_NAME,
        Path(__file__).parent.parent.absolute()
        / Path("experiments/configs/data_generator_configs"),
    )
    data_generator = get_data_generator(data_config)

    second_move_checkmates = 0

    for index in range(NUM_POSITIONS):
        if index % 10000 == 0:
            print(f"Scanned {index}/{NUM_POSITIONS} boards")
        board = data_generator.next()

        # Compute the only legal move
        legal_moves = list(board.legal_moves)
        assert len(legal_moves) == 1
        move = legal_moves[0]

        # Make the move and check whether the game is over
        board.push(move)
        if board.is_game_over():
            second_move_checkmates += 1

    print(
        f"Found {second_move_checkmates} second move checkmates out of {NUM_POSITIONS} positions."
    )
