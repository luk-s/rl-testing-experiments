import asyncio
from pathlib import Path
from typing import List, Union, overload

import chess
import chess.engine
from board_analysis import analyze_with_engine
from chess.engine import Cp, Score
from load_results import compute_differences, flip_q_values, load_data

from rl_testing.config_parsers import get_engine_config
from rl_testing.engine_generators import get_engine_generator
from rl_testing.util.util import cp2q, q2cp

STOCKFISH_PATH = "/home/lukas/Software/stockfish/stockfish_15_linux_x64_avx2/stockfish_15_x64_avx2"


@overload
def score2float(score: List[Union[Score, float]]) -> List[float]:
    ...


@overload
def score2float(score: Union[Score, float]) -> float:
    ...


def score2float(
    score: Union[List[Union[Score, float]], Union[Score, float]]
) -> Union[float, List[float]]:
    if isinstance(score, list):
        return [score2float(s) for s in score]
    if isinstance(score, Score):
        return score.score() / 1.0
    elif isinstance(score, float):
        return score
    else:
        raise TypeError(f"Unknown type {type(score)}")


def find_better_evaluations(
    leela_scores1: List[Score],
    leela_scores2: List[Score],
    stockfish_scores: Union[List[Score], List[int]],
) -> List[str]:
    better = []
    for leela1, leela2, stockfish in zip(leela_scores1, leela_scores2, stockfish_scores):
        if isinstance(stockfish, int):
            stockfish = Cp(stockfish)
        assert leela1 != leela2
        if stockfish == leela1:
            better.append("network1")
        elif stockfish == leela2:
            better.append("network2")
        elif stockfish < leela1 < leela2:
            better.append("network1")
        elif leela1 < stockfish < leela2:
            leela1, leela2, stockfish = score2float([leela1, leela2, stockfish])
            if abs(stockfish - leela1) < abs(stockfish - leela2):
                better.append("network1")
            elif abs(stockfish - leela1) > abs(stockfish - leela2):
                better.append("network2")
        elif leela1 < leela2 < stockfish:
            better.append("network2")
        elif stockfish < leela2 < leela1:
            better.append("network2")
        elif leela2 < stockfish < leela1:
            leela1, leela2, stockfish = score2float([leela1, leela2, stockfish])
            if abs(stockfish - leela1) < abs(stockfish - leela2):
                better.append("network1")
            elif abs(stockfish - leela1) > abs(stockfish - leela2):
                better.append("network2")
        elif leela2 < leela1 < stockfish:
            better.append("network1")
        else:
            raise ValueError("This shouldn't happen!")

    return better


if __name__ == "__main__":
    result_folder = Path(__file__).parent.parent / Path(
        # "results/differential_testing/main_experiment/results_fixed_and_long"
        "results/differential_testing/main_results"
        # "results/forced_moves/main_experiment"
    )
    # result_file = Path("results_ENGINE_local_5000_nodes_DATA_random_fen_database.txt")
    # result_file = Path("results_ENGINE_local_400_nodes_DATA_forced_moves_fen_database.txt")
    # result_file = Path("results_ENGINE_local_400_nodes_DATA_late_move_fen_database.txt")
    result_file = Path("results_ENGINE_local_dag_1_node_DATA_database.txt")
    num_largest = 100
    fen_key = "fen"
    score_key = "score"
    move_key = "best_move"
    q_vals_to_flip = []  # ["Q2"]
    engine_config_name = "remote_25_depth_stockfish.ini"  # "remote_400_nodes.ini"
    network_name = ""  # "network_600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2"
    search_limit = {"depth": 25}

    dataframe, _ = load_data(result_folder / result_file)
    for column_name in q_vals_to_flip:
        dataframe = flip_q_values(dataframe, column_name=column_name)
    dataframe = compute_differences(
        dataframe=dataframe, column_name1=score_key + "1", column_name2=score_key + "2"
    )

    interesting_boards = []
    leela_cp_scores1, leela_cp_scores2 = [], []
    leela_q_scores1, leela_q_scores2 = [], []
    differences = []
    for i in range(num_largest):
        fen = dataframe.iloc[i][fen_key]
        fen = fen.replace("_", " ")
        interesting_boards.append(chess.Board(fen))
        difference = dataframe.iloc[i]["difference"]
        leela_cp_scores1.append(q2cp(dataframe.iloc[i][score_key + "1"]))
        leela_cp_scores2.append(q2cp(dataframe.iloc[i][score_key + "2"]))
        leela_q_scores1.append(dataframe.iloc[i][score_key + "1"])
        leela_q_scores2.append(dataframe.iloc[i][score_key + "2"])
        differences.append(difference)

    # Analyze the positions with stockfish
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())

    # Setup engine generator
    engine_config = get_engine_config(
        config_name=engine_config_name,
        config_folder_path=Path(__file__).parent.parent.absolute()
        / Path("configs/engine_configs/"),
    )
    engine_generator = get_engine_generator(engine_config)
    if network_name:
        engine_generator.set_network(network_name)

    stockfish_cp_scores, trees = asyncio.run(
        analyze_with_engine(
            engine_generator=engine_generator,
            positions=interesting_boards,
            search_limits=search_limit,
        )
    )
    stockfish_q_scores = [cp2q(score) for score in stockfish_cp_scores]

    # Decide which evaluations are better
    better = find_better_evaluations(
        leela_scores1=leela_q_scores1,
        leela_scores2=leela_q_scores2,
        stockfish_scores=stockfish_q_scores,
    )

    network1_better = better.count("network1")
    network2_better = better.count("network2")
    unknown = better.count("unknown")

    for index, board in enumerate(interesting_boards):
        fen = board.fen(en_passant="fen")
        cp1 = str(leela_cp_scores1[index])
        cp2 = str(leela_cp_scores2[index])
        q1 = f"{leela_q_scores1[index]: .3f}"
        q2 = f"{leela_q_scores2[index]: .3f}"
        stockfish_cp = str(stockfish_cp_scores[index])
        stockfish_q = f"{stockfish_q_scores[index]: .3f}"
        result = better[index]
        suffix = " is better" if result != "unknown" else ""
        print(
            # f"{fen :<74} Q1: {q1:<9} Q2: {q2:<9} CP1: {cp1:<7} CP2: {cp2:<7}, "
            f"{fen :<74} Q1: {q1:<9} Q2: {q2:<9}, "
            f"Diff: {differences[index] :8.5f} "
            f"Stockfish: {stockfish_q :<7} Result: {result + suffix}"
        )
    print()
    print(f"Network 1 was {network1_better}/{len(better)} times better")
    print(f"Network 2 was {network2_better}/{len(better)} times better")
    print(f"{unknown}/{len(better)} times it's not clear")
