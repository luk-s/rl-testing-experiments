from pathlib import Path

from rl_testing.config_parsers import BoardGeneratorConfig
from rl_testing.data_generators import RandomBoardGenerator

if __name__ == "__main__":
    DATA_CONFIG_NAME = "random_many_pieces.ini"
    NUM_POSITIONS_TO_CREATE = 100000
    FILE_NAME = "data/random_positions.txt"

    BoardGeneratorConfig.set_config_folder_path(
        Path(__file__).parent.parent.absolute()
        / Path("experiments/configs/data_generator_configs")
    )
    data_config = BoardGeneratorConfig.from_config_file(DATA_CONFIG_NAME)
    data_generator = RandomBoardGenerator(**data_config.board_generator_config)

    with open(FILE_NAME, "a") as f:
        for i in range(NUM_POSITIONS_TO_CREATE):
            print(f"Creating board {i}/{NUM_POSITIONS_TO_CREATE}")
            fen = data_generator.next().fen(en_passant="fen")
            f.write(fen + "\n")
