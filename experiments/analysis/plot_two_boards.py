from pathlib import Path

import chess
import chess.svg
import pandas as pd

from rl_testing.util.chess import plot_board, plot_two_boards

result_path = Path(__file__).absolute().parent / "results/examiner_testing/main_experiment"
file_path = "results_VICTIM_ENGINE_local_400_nodes_EXAMINER_ENGINE_local_25_depth_stockfish_DATA_database.txt"

board = chess.Board("8/8/8/8/8/5Kp1/6pk/1R6 b - - 1 58")

print(list(board.legal_moves))

move = chess.Move.from_uci("g2g1q")
move2 = chess.Move.from_uci("e2e4")

# open database
# data = pd.read_csv(result_path / file_path, sep=",")
#
# filtered_data = data[data["success"] == 1]
#
# for row in filtered_data.itertuples():
#     print("\nROW")
#     print(f"FEN1: {row.board_original}")
#     print(f"FEN2: {row.board_adversarial}")
#     print(f"Leela eval 1: {row.score_original_v}")
#     print(f"Leela eval 2: {row.score_adversarial_v}")
#     print(f"Score difference: {abs(row.score_original_v - row.score_adversarial_v)}")
#     print(f"Stockfish eval 1: {row.score_original_e}")
#     print(f"Stockfish eval 2: {row.score_adversarial_e}")


print("Finished!")

# Failure case 1
# board1 = chess.Board("3r2k1/1b3p2/p3q1pp/1p6/3B2n1/QP2PpP1/P6P/2RN1BK1 b - - 3 34")
# board2 = chess.Board("6k1/1b3p2/p3q1pp/1p6/3r2n1/QP2PpP1/P6P/2RN1BK1 w - - 0 35")

# Failure case 2
board1 = chess.Board("r5k1/r5Pp/8/1q2p3/4Q3/1p6/p2R4/K5R1 b - - 2 39")
board2 = chess.Board("4r1k1/r5Pp/8/1q2p3/4Q3/1p6/p2R4/K5R1 w - - 3 40")

# plot_board(
#     board1,
#     title="Board 1: Black to play",
#     fontsize=18,
#     save_path="board1.png",
#     x_label="Win probability:    65% for Black\nBest move:           Re8                ",
#     arrows=[chess.svg.Arrow(chess.A8, chess.E8, color="green")],
#    save=True,
# )

# plot_board(
#     board2,
#     title="Board 2: White to play",
#     fontsize=18,
#     save_path="board2.png",
#     x_label="Win probability:   0% for Black\n ",
#     save=True,
# )

plot_two_boards(
    board1=board1,
    board2=board2,
    title1="Board 1: Black to move",
    title2="Board 2: White to move",
    x_label1="Win probability:           68% for Black\nBest move:                  Re8",
    x_label2="Win probability:            0% for Black",
    fontsize=14,
    plot_size=800,
    save=True,
    show=False,
    save_path="combined_auto.png",
    arrows1=[chess.svg.Arrow(chess.A8, chess.E8, color="green")],
    arrows2=[],
)
