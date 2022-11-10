import asyncio
from pathlib import Path
from typing import List

import chess
import chess.engine
import numpy as np
from chess.engine import Cp, Score

from load_results import compute_differences, flip_q_values, load_data

STOCKFISH_PATH = "/home/lukas/Software/stockfish/stockfish_15_linux_x64_avx2/stockfish_15_x64_avx2"


async def analyze_with_stockfish(
    stockfish_path: str, positions: List[chess.Board], search_depth: int = 30
) -> List[Score]:
    stockfish_scores = []
    _, engine = await chess.engine.popen_uci(stockfish_path)
    await engine.configure({"Threads": 1})
    for board_index, board in enumerate(positions):
        fen = board.fen(en_passant="fen")
        print(f"Analyzing board {board_index+1}/{len(positions)}: {fen}")
        info = await engine.analyse(board, chess.engine.Limit(depth=search_depth))
        # stockfish_scores.append(info["score"].white())
        stockfish_scores.append(info["score"].relative)

    await engine.quit()

    return stockfish_scores


def q2cp(q_value):
    return Cp(round(111.714640912 * np.tan(1.5620688421 * q_value)))


def find_better_evaluations(
    leela_scores1: List[Score], leela_scores2: List[Score], stockfish_scores: List[Score]
) -> List[str]:
    better = []
    for leela1, leela2, stockfish in zip(leela_scores1, leela_scores2, stockfish_scores):
        assert leela1 != leela2
        if stockfish == leela1:
            better.append("network1")
        elif stockfish == leela2:
            better.append("network2")
        elif stockfish < leela1 < leela2:
            better.append("network1")
        elif leela1 < stockfish < leela2:
            if isinstance(stockfish, Cp):
                if abs(stockfish.score() - leela1.score()) < abs(
                    stockfish.score() - leela2.score()
                ):
                    better.append("network1")
                elif abs(stockfish.score() - leela1.score()) > abs(
                    stockfish.score() - leela2.score()
                ):
                    better.append("network2")
            else:
                better.append("unknown")
        elif leela1 < leela2 < stockfish:
            better.append("network2")
        elif stockfish < leela2 < leela1:
            better.append("network2")
        elif leela2 < stockfish < leela1:
            if isinstance(stockfish, Cp):
                if abs(stockfish.score() - leela1.score()) < abs(
                    stockfish.score() - leela2.score()
                ):
                    better.append("network1")
                elif abs(stockfish.score() - leela1.score()) > abs(
                    stockfish.score() - leela2.score()
                ):
                    better.append("network2")
            else:
                better.append("unknown")
        elif leela2 < leela1 < stockfish:
            better.append("network1")
        else:
            raise ValueError("This shouldn't happen!")

    return better


if __name__ == "__main__":
    result_folder = Path(__file__).parent.parent / Path(
        "results/differential_testing/main_experiment"
        # "results/forced_moves/main_experiment"
    )
    result_file = Path("results_ENGINE_local_5000_nodes_DATA_random_fen_database.txt")
    # result_file = Path("results_ENGINE_local_400_nodes_DATA_forced_moves_fen_database.txt")
    # result_file = Path("results_ENGINE_local_400_nodes_DATA_late_move_fen_database.txt")
    # result_file = Path("results_ENGINE_local_1_node_DATA_late_move_fen_database.txt")
    num_largest = 100
    fen_key = "FEN"
    q_vals_to_flip = []  # ["Q2"]
    search_depth = 20

    dataframe, _ = load_data(result_folder / result_file)
    for column_name in q_vals_to_flip:
        dataframe = flip_q_values(dataframe, column_name=column_name)
    dataframe = compute_differences(dataframe=dataframe, column_name1="Q1", column_name2="Q2")

    interesting_boards = []
    leela_cp_scores1 = []
    leela_cp_scores2 = []
    differences = []
    for i in range(num_largest):
        fen = dataframe.iloc[i][fen_key]
        fen = fen.replace("_", " ")
        interesting_boards.append(chess.Board(fen))
        difference = dataframe.iloc[i]["difference"]
        leela_cp_scores1.append(q2cp(dataframe.iloc[i]["Q1"]))
        leela_cp_scores2.append(q2cp(dataframe.iloc[i]["Q2"]))
        differences.append(difference)

    # Analyze the positions with stockfish
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    stockfish_scores = asyncio.run(
        analyze_with_stockfish(
            stockfish_path=STOCKFISH_PATH, positions=interesting_boards, search_depth=search_depth
        )
    )

    # Decide which evaluations are better
    better = find_better_evaluations(
        leela_scores1=leela_cp_scores1,
        leela_scores2=leela_cp_scores2,
        stockfish_scores=stockfish_scores,
    )

    network1_better = better.count("network1")
    network2_better = better.count("network2")
    unknown = better.count("unknown")

    for index, board in enumerate(interesting_boards):
        fen = board.fen(en_passant="fen")
        q1 = str(leela_cp_scores1[index])
        q2 = str(leela_cp_scores2[index])
        stockfish = str(stockfish_scores[index])
        result = better[index]
        suffix = " is better" if result != "unknown" else ""
        print(
            f"{fen :<74} CP1: {q1:<7} CP2: {q2:<7}, "
            # f"Difference: {differences[index] :<20} CP1: {q1:<7} CP2: {q2:<7}, "
            f"Stockfish: {stockfish :<7} Result: {result + suffix}"
        )
    print()
    print(f"Network 1 was {network1_better}/{len(better)} times better")
    print(f"Network 2 was {network2_better}/{len(better)} times better")
    print(f"{unknown}/{len(better)} times it's not clear")
