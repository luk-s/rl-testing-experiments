import argparse
import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import chess.engine
import numpy as np
from chess.engine import Cp, Score

from rl_testing.config_parsers import get_data_generator_config, get_engine_config
from rl_testing.data_generators import BoardGenerator, get_data_generator
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.engine_generators.relaxed_uci_protocol import RelaxedUciProtocol
from rl_testing.util.util import MoveStat, PositionStat, parse_info, plot_board

RESULT_DIR = Path(__file__).parent / Path("results/differential_testing")


async def analyze_positions(
    queue: asyncio.Queue,
    engine: RelaxedUciProtocol,
    search_limits: Dict[str, Any],
    num_boards: int,
    sleep_after_get: float = 0.0,
    engine_generator: Optional[EngineGenerator] = None,
) -> List[Union[Score, str]]:

    stockfish_scores = []

    # Iterate over all boards
    for board_index in range(num_boards):
        # Fetch the next board from the queue
        board = await queue.get()

        await asyncio.sleep(delay=sleep_after_get)

        print(f"Analyzing board {board_index + 1}/{num_boards}: " + board.fen(en_passant="fen"))

        # Needs to be in a try-except because the engine might crash unexpectedly
        try:
            # Start the analysis
            info = await engine.analyse(board, chess.engine.Limit(**search_limits))
            stockfish_scores.append(info["score"].relative)

        except chess.engine.EngineTerminatedError:
            if engine_generator is None:
                print("Can't restart engine due to missing generator")
                raise

            # Mark the current board as failed
            stockfish_scores.append("invalid")

            # Try to restart the engine
            print("Trying to restart engine")

            engine = await engine_generator.get_initialized_engine(initialize_network=False)
        finally:
            queue.task_done()

    return stockfish_scores


async def get_positions_async(
    queue: asyncio.Queue,
    data_generator: BoardGenerator,
    num_positions: int = 1,
    sleep_between_positions: float = 0.05,
) -> List[chess.Board]:
    boards = []

    # Create random chess positions if necessary
    for board_index in range(num_positions):

        # Create a random chess position
        board_candidate = data_generator.next()

        # Check if the generated position was valid
        if board_candidate != "failed":
            boards.append(board_candidate)
            fen = board_candidate.fen(en_passant="fen")
            print(f"Created board {board_index}: " f"{fen}")
            await queue.put(board_candidate.copy())

            await asyncio.sleep(delay=sleep_between_positions)

    return boards


async def analyze_positions_stockfish(
    engine_generator: EngineGenerator,
    data_generator: BoardGenerator,
    *,
    search_limits: Optional[Dict[str, Any]] = None,
    num_positions: int = 1,
    sleep_between_positions: float = 0.1,
) -> Tuple[List[chess.Board], List[Tuple[float, float]]]:

    engine = await engine_generator.get_initialized_engine(initialize_network=False)

    # If no search limit has been specified, just search one single node
    if search_limits is None:
        search_limits = {"depth": 25}

    scores = []
    boards_final = []
    queue = asyncio.Queue()

    data_generator_task = asyncio.create_task(
        get_positions_async(
            queue=queue,
            data_generator=data_generator,
            num_positions=num_positions,
            sleep_between_positions=sleep_between_positions,
        )
    )

    data_consumer_task = asyncio.create_task(
        analyze_positions(
            queue=queue,
            engine=engine,
            search_limits=search_limits,
            num_boards=num_positions,
            sleep_after_get=0.001,
            engine_generator=engine_generator,
        )
    )

    # Wait until all tasks finish
    boards, scores_unfiltered = await asyncio.gather(data_generator_task, data_consumer_task)

    # await engine1.quit()
    # await engine2.quit()

    for board_index, board in enumerate(boards):

        if scores_unfiltered[board_index] != "invalid":
            try:
                scores.append(scores_unfiltered[board_index])
                boards_final.append(board)
            except:
                print(f"ERROR parsing the results of board {board_index}")
                print("ERROR BOARD: ", board.fen(en_passant="fen"))
        else:
            message = "Analysis of board: " + board.fen(en_passant="fen") + " is invalid!"
            print(message)

    return boards_final, scores


def analyze_results(
    boards: List[chess.Board],
    results: List[Tuple[float, float]],
    difference_threshold: float = 1.0,
    show: bool = True,
    result_dir: Union[str, Path] = "",
    fontsize: int = 22,
) -> Tuple[chess.Board, float, float, float]:
    # Find results where the models have a large difference
    results = np.array(results)
    differences = np.abs(results[:, 0] - results[:, 1])
    indices = np.where(differences >= difference_threshold)[0]

    special_results = []

    for index in indices:
        board = boards[index]
        value1, value2 = results[index]
        special_results.append((board, differences[index], value1, value2))
        title = f"value1: {value1}, value2: {value2}"
        fen = board.fen(en_passant="fen")
        fen = fen.replace(" ", "_")
        fen = fen.replace("/", "|")
        file_path = Path(result_dir) / Path(fen + ".png")
        plot_board(board, title=title, fen=fen, show=show, save_path=file_path, fontsize=fontsize)

    return special_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################
    #           CONFIG START         #
    ##################################
    # NETWORKS:
    # =========
    # strong and recent: "network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42"
    # from leela paper: "network_c8368caaccd43323cc513465fb92740ea6d10b50684639a425fca2b42fc1f7be"
    # strong recommended: "network_600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2" # noqa: E501
    # Weak local 1: "f21ee51844a7548c004a1689eacd8b4cd4c6150d6e03c732b211cf9963d076e1"
    # Weak local 2: "fbd5e1c049d5a46c098f0f7f12e79e3fb82a7a6cd1c9d1d0894d0aae2865826f"

    # fmt: off
    parser.add_argument("--seed",               type=int, default=42)
    parser.add_argument("--engine_config_name", type=str, default="local_25_depth_stockfish.ini")
    parser.add_argument("--data_config_name",   type=str, default="late_move_fen_database.ini")
    parser.add_argument("--num_positions",      type=int, default=100_000)
    parser.add_argument("--result_subdir",      type=str, default="stockfish_analysis")
    # fmt: on
    ##################################
    #           CONFIG END           #
    ##################################
    args = parser.parse_args()

    np.random.seed(args.seed)

    engine_config = get_engine_config(
        config_name=args.engine_config_name,
        config_folder_path=Path(__file__).parent.absolute() / Path("configs/engine_configs/"),
    )
    engine_generator = get_engine_generator(engine_config)

    data_config = get_data_generator_config(
        config_name=args.data_config_name,
        config_folder_path=Path(__file__).parent.absolute()
        / Path("configs/data_generator_configs"),
    )
    data_generator = get_data_generator(data_config)

    # Run the differential testing
    start_time = time.perf_counter()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    boards, scores = asyncio.run(
        analyze_positions_stockfish(
            engine_generator=engine_generator,
            data_generator=data_generator,
            search_limits=engine_config.search_limits,
            num_positions=args.num_positions,
        )
    )

    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time: .3f} seconds")

    # Create results-file-name
    engine_config_name = args.engine_config_name[:-4]
    data_config_name = args.data_config_name[:-4]

    # Open the results file
    result_directory = RESULT_DIR / args.result_subdir
    result_directory.mkdir(parents=True, exist_ok=True)
    with open(
        result_directory
        / Path(f"results_ENGINE_{engine_config_name}_DATA_{data_config_name}.txt"),
        "a",
    ) as f:
        # Store the config
        f.write(f"SEED = {args.seed}\n")
        f.write(f"ENGINE_CONFIG_NAME = {args.engine_config_name}\n")
        f.write(f"DATA_CONFIG_NAME = {args.data_config_name}\n")
        f.write(f"NUM_POSITIONS = {args.num_positions}\n")
        f.write(f"RESULT_SUBDIR = {args.result_subdir}\n")
        f.write("\n")

        # Store the results
        f.write("FEN,Score\n")
        for board, score in zip(boards, scores):
            fen = board.fen(en_passant="fen").replace(" ", "_")
            f.write(f"{fen},{str(score)}\n")
