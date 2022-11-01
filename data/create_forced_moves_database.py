from pathlib import Path

from rl_testing.config_parsers import BoardGeneratorConfig
from rl_testing.data_generators import DataBaseBoardGenerator

if __name__ == "__main__":
    DATA_CONFIG_NAME = "database_late_moves.ini"
    NUM_POSITIONS_TO_CREATE = 100000
    FILE_NAME = "data/forced_move_positions.txt"

    BoardGeneratorConfig.set_config_folder_path(
        Path(__file__).parent.parent.absolute()
        / Path("experiments/configs/data_generator_configs")
    )
    data_config = BoardGeneratorConfig.from_config_file(DATA_CONFIG_NAME)
    data_generator = DataBaseBoardGenerator(**data_config.board_generator_config)

    with open(FILE_NAME, "a") as f:
        boards_read = 0
        for i in range(NUM_POSITIONS_TO_CREATE):
            while True:
                if boards_read % 10000 == 0:
                    print(f"Scanned {boards_read} boards")
                board = data_generator.next()
                boards_read += 1
                if board.legal_moves.count() == 1:
                    break

            print(
                f"Found forced move {i+1}/{NUM_POSITIONS_TO_CREATE} "
                f"after scanning {boards_read} boards."
            )

            f.write(board.fen(en_passant="fen") + "\n")
