from pathlib import Path

import chess
import chess.pgn
from rl_testing.data_generators.generators import BoardGenerator

DATA_PATH = data_path = Path(__file__).parent.parent.parent


class DataBaseBoardGenerator(BoardGenerator):
    def __init__(self, database_name: str, open_now: bool = True):
        self.data_path = DATA_PATH / database_name
        self.file_iterator = None

        self.current_game = None
        self.current_board = None
        self.games_read = 0
        self.moves_read = 0

        if open_now:
            self.setup_position()

    def setup_position(self) -> None:
        self.file_iterator = open(self.data_path, "r")

        # Read all games which have already been read
        for _ in range(self.games_read):
            self.current_game = chess.pgn.read_game(self.file_iterator)

        if self.current_game is None:
            self.current_game = chess.pgn.read_game(self.file_iterator)

        self.current_board = chess.Board()
        moves = list(self.current_game.mainline_moves())

        # Read all moves which have already been read
        for move_index, move in enumerate(moves):
            if move_index >= self.moves_read:
                break
            self.current_board.push(move=move)

    def next(self) -> chess.Board:
        # If the file has been closed, re-open it
        if self.file_iterator is None:
            self.setup_position()

        moves = self.current_game.mainline_moves()

        # If all moves of this game have been read, load a new game
        # Because some games are empty, repeat this process until you
        # find a game which is not empty
        while self.moves_read >= len(moves):
            self.games_read += 1
            self.moves_read = 0
            self.current_game = chess.pgn.read_game(self.file_iterator)
            self.current_board = chess.Board()
            moves = self.current_game.mainline_moves()

        # Read the next move
        self.current_board.push(moves[self.moves_read])
        self.moves_read += 1

        # Return the new position
        return self.current_board


if __name__ == "__main__":
    """
    TODO:
        1. Test implementation
        2. Add a parameter which only reads in moves after a certain depth
    """
