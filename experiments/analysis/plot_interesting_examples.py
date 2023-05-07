import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import chess.engine
import pandas as pd
from chess.engine import Score

from rl_testing.config_parsers import get_engine_config
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.mcts.tree_parser import (
    NodeInfo,
    TreeInfo,
    convert_tree_to_networkx,
    plot_networkx_tree,
)
from rl_testing.util.chess import cp2q, plot_two_boards, transform_board_to_png


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, help="Path to result file which stores the results that should be analyzed", required=True)  # noqa
    parser.add_argument("--score_type1", type=str, help="Whether the score of the root node should be extracted or the score of the best move for each first position", required=False, choices=["node", "best_move"], default="best_move")  # noqa
    parser.add_argument("--score_type2", type=str, help="Whether the score of the root node should be extracted or the score of the best move for each second position", required=False, choices=["node", "best_move"], default="node")  # noqa
    parser.add_argument("--num_examples", type=int, help="Number of examples to plot", required=False, default=10)  # noqa
    parser.add_argument("--engine_config_name", type=str, help="Name of the engine config to use", required=True)  # noqa
    parser.add_argument("--network_name", type=str, default="T807785-b124efddc27559564d6464ba3d213a8279b7bd35b1cbfcf9c842ae8053721207")  # noqa
    parser.add_argument("--fen_column1", type=str, help="Name of the column storing the first fen value", required=False, default="parent_fen")  # noqa
    parser.add_argument("--fen_column2", type=str, help="Name of the column storing the second fen value", required=False, default="child_fen")  # noqa
    parser.add_argument("--flip_second_score", action="store_true", help="Whether the second Q-value should be flipped (multiplied by -1)", required=False, default=False)  # noqa
    parser.add_argument("--show_best_move_first", action="store_true", help="Whether the best move should be shown for the first position", required=False, default=False)  # noqa
    parser.add_argument("--show_best_move_second", action="store_true", help="Whether the best move should be shown for the second position", required=False, default=False)  # noqa
    parser.add_argument("--save_plot", action="store_true", help="Save the resulting plots to a file", required=False, default=False)  # noqa
    parser.add_argument("--save_path_base", type=str, help="Base path to use for saving the plots", required=False, default=None)  # noqa
    parser.add_argument("--show_plot", action="store_true", help="Show the resulting plots", required=False, default=False)  # noqa
    # fmt: on
    return parser.parse_args()


async def analyze_with_engine(
    engine_generator: EngineGenerator,
    positions: List[Union[chess.Board, str]],
    network_name: Optional[str] = None,
    search_limits: Optional[Dict[str, Any]] = None,
    score_type: str = "cp",
) -> Tuple[List[Score], List[TreeInfo]]:
    valid_score_types = ["cp", "q"]
    engine_scores = []
    verbose_stats = []

    # Set search limits
    if search_limits is None:
        search_limits = {"depth": 25}

    # Setup and configure the engine
    if network_name is not None:
        engine_generator.set_network(network_name)
    engine = await engine_generator.get_initialized_engine()

    for board_index, board in enumerate(positions):
        # Make sure that 'board' has type 'chess.Board'
        if isinstance(board, str):
            fen = board
            board = chess.Board(fen=fen)
        else:
            fen = board.fen(en_passant="fen")

        # Analyze the position
        print(f"Analyzing board {board_index+1}/{len(positions)}: {fen}")
        info = await engine.analyse(board, chess.engine.Limit(**search_limits))

        # Extract the score
        cp_score = info["score"].relative.score(mate_score=12780)
        if score_type == "cp":
            engine_scores.append(cp_score)
        elif score_type == "q":
            engine_scores.append(cp2q(cp_score))
        else:
            raise ValueError(
                f"Invalid score type: {score_type}. Choose one from {valid_score_types}"
            )

        if "root_and_child_scores" in info:
            verbose_stats.append(info["root_and_child_scores"])

    return engine_scores, verbose_stats


def analyze_positions(
    fens: List[str],
    args: argparse.Namespace,
) -> List[NodeInfo]:
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())

    # Setup engine generator
    engine_config = get_engine_config(
        config_name=args.engine_config_name,
        config_folder_path=Path(__file__).parent.parent.absolute()
        / Path("configs/engine_configs/"),
    )
    search_limit = engine_config.search_limits
    engine_generator = get_engine_generator(engine_config)

    _, verbose_stats = asyncio.run(
        analyze_with_engine(
            engine_generator=engine_generator,
            positions=fens,
            network_name=args.network_name,
            search_limits=search_limit,
        )
    )

    return verbose_stats


def extract_win_prob_and_best_move_from_verbose_stat(
    verbose_stat: NodeInfo, mode: str, flip_q_value: bool = False
) -> Tuple[float, chess.Move]:
    valid_modes = ["node", "best_move"]
    assert mode in valid_modes, f"Invalid mode: {mode}. Choose one from {valid_modes}"

    # Get the best move
    best_edge = max(verbose_stat.child_edges, key=lambda s: s.num_visits + s.in_flight_visits)
    best_move = best_edge.move

    # Extract the win probability
    if mode == "node":
        q_value = verbose_stat.q_value
        d_value = verbose_stat.draw_value
    elif mode == "best_move":
        q_value = best_edge.q_value
        d_value = best_edge.draw_value

    if flip_q_value:
        q_value *= -1

    win_prob = 0.5 * (q_value + 1 - d_value)

    return win_prob, best_move


def create_two_board_plot(
    first_fen: str,
    second_fen: str,
    first_win_prob: float,
    second_win_prob: float,
    second_win_prob_flipped: float,
    best_move_first: Optional[chess.Move] = None,
    best_move_second: Optional[chess.Move] = None,
    show_plot: bool = True,
    save_plot: bool = False,
    save_path: str = "",
):
    if save_plot:
        assert save_path, "If save_plot is True, save_path must be specified!"

    # Create the boards
    first_board = chess.Board(first_fen)
    second_board = chess.Board(second_fen)

    # Extract the colors of the players to move
    first_player_to_move = "White" if first_board.turn == chess.WHITE else "Black"
    second_player_to_move = "White" if second_board.turn == chess.WHITE else "Black"
    first_color = first_player_to_move
    if second_win_prob_flipped:
        second_color = "White" if second_player_to_move == "Black" else "Black"
    else:
        second_color = second_player_to_move

    # Create the titles
    title1 = f"Board 1: {first_player_to_move} to move"
    title2 = f"Board 2: {second_player_to_move} to move"

    first_win_prob = str(round(100 * first_win_prob))
    second_win_prob = str(round(100 * second_win_prob))

    # Compute the number of digits in the win probabilities
    first_len = len(str(first_win_prob))
    second_len = len(str(second_win_prob))

    # Compute the correct amount of spaces to add
    first_spaces = " " * (8 - first_len)
    second_spaces = " " * (8 - second_len)

    # Create the x-axis labels
    x_label1 = f"Win probability:{first_spaces}{first_win_prob}% for {first_color}"  # 16
    if best_move_first:
        x_label1 += f"\nBest move:{first_spaces + ' ' * 6}{str(best_move_first)}"  # 10

    x_label2 = f"Win probability:{second_spaces}{second_win_prob}% for {second_color}"
    if best_move_second:
        x_label2 += f"\nBest move:{}{str(best_move_second)}"

    # Build the arrows of the best moves
    arrows1 = []
    arrows2 = []
    if best_move_first:
        arrows1 = [
            chess.svg.Arrow(best_move_first.from_square, best_move_first.to_square, color="green")
        ]
    if best_move_second:
        arrows2 = [
            chess.svg.Arrow(best_move_second.from_square, best_move_second.to_square, color="green")
        ]

    # Create the plots
    plot_two_boards(
        board1=first_board,
        board2=second_board,
        arrows1=arrows1,
        arrows2=arrows2,
        title1=title1,
        title2=title2,
        x_label1=x_label1,
        x_label2=x_label2,
        fontsize=14,
        plot_size=800,
        save=save_plot,
        show=show_plot,
        save_path=save_path,
    )


def plot_interesting_examples(args: argparse.Namespace):
    # Load the data
    dataframe = pd.read_csv(args.result_path)

    # Make sure that the dataframe is sorted by the difference column
    dataframe = dataframe.sort_values(by="difference", ascending=False)

    # Extract the FENs
    first_fens = dataframe[[args.fen_column1]].values.transpose().tolist()[0]
    second_fens = dataframe[[args.fen_column2]].values.transpose().tolist()[0]

    # Extract only the first args.num_examples
    first_fens = first_fens[: args.num_examples]
    second_fens = second_fens[: args.num_examples]

    # Analyze the positions
    all_fens = first_fens + second_fens
    all_stats = analyze_positions(all_fens, args)
    first_stats, second_stats = (
        all_stats[: args.num_examples],
        all_stats[args.num_examples :],
    )

    # Extract the win probabilities
    first_win_probs_and_best_moves = [
        extract_win_prob_and_best_move_from_verbose_stat(stat, args.score_type1)
        for stat in first_stats
    ]
    second_win_probs_and_best_moves = [
        extract_win_prob_and_best_move_from_verbose_stat(
            stat, args.score_type2, args.flip_second_score
        )
        for stat in second_stats
    ]
    first_win_probs, first_best_moves = zip(*first_win_probs_and_best_moves)
    second_win_probs, second_best_moves = zip(*second_win_probs_and_best_moves)

    # Create the plots
    for index, (
        first_fen,
        second_fen,
        first_win_prob,
        second_win_prob,
        first_best_move,
        second_best_move,
    ) in enumerate(
        zip(
            first_fens,
            second_fens,
            first_win_probs,
            second_win_probs,
            first_best_moves,
            second_best_moves,
        )
    ):
        save_path = str(args.save_path_base) + f"_{index+1}.png"
        print(f"Plotting example {index+1}/{args.num_examples}")
        create_two_board_plot(
            first_fen=first_fen,
            second_fen=second_fen,
            first_win_prob=first_win_prob,
            second_win_prob=second_win_prob,
            second_win_prob_flipped=args.flip_second_score,
            best_move_first=first_best_move if args.show_best_move_first else None,
            best_move_second=second_best_move if args.show_best_move_second else None,
            show_plot=args.show_plot,
            save_plot=args.save_plot,
            save_path=save_path,
        )


if __name__ == "__main__":
    # Get all command-line arguments
    args = parse_args()

    # Run the analysis
    plot_interesting_examples(args)
