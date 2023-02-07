import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import chess.engine
import numpy as np

from rl_testing.config_parsers import get_data_generator_config, get_engine_config
from rl_testing.data_generators import BoardGenerator, get_data_generator
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.engine_generators.relaxed_uci_protocol import RelaxedUciProtocol
from rl_testing.util.chess import plot_board
from rl_testing.util.experiment import store_experiment_params
from rl_testing.util.util import get_task_result_handler

RESULT_DIR = Path(__file__).parent / Path("results/differential_testing")


async def get_positions_async(
    queue: asyncio.Queue,
    data_generator: BoardGenerator,
    num_positions: int = 1,
    sleep_between_positions: float = 0.05,
) -> None:
    boards = []

    # Create random chess positions if necessary
    for board_index in range(num_positions):

        # Create a random chess position
        board_candidate = data_generator.next()

        # Check if the generated position was valid
        if board_candidate != "failed":
            boards.append(board_candidate)
            fen = board_candidate.fen(en_passant="fen")
            logging.info(f"Created board {board_index}: " f"{fen}")
            await queue.put(board_candidate.copy())

            await asyncio.sleep(delay=sleep_between_positions)


async def analyze_positions(
    queue: asyncio.Queue,
    engine: RelaxedUciProtocol,
    search_limits: Dict[str, Any],
    num_boards: int,
    file_path: Union[str, Path],
    sleep_after_get: float = 0.0,
    engine_generator: Optional[EngineGenerator] = None,
) -> None:
    # Required to ensure that the engine doesn't use cached results from
    # previous analyses
    analysis_counter = 0

    with open(file_path, "a") as file:
        csv_header = "fen,score,best_move\n"
        file.write(csv_header)

        # Iterate over all boards
        for board_index in range(num_boards):
            # Fetch the next board from the queue
            board = await queue.get()

            await asyncio.sleep(delay=sleep_after_get)

            logging.info(
                f"Analyzing board {board_index + 1}/{num_boards}: " + board.fen(en_passant="fen")
            )

            # Needs to be in a try-except because the engine might crash unexpectedly
            try:
                # Start the analysis
                analysis_counter += 1
                info = await engine.analyse(
                    board, chess.engine.Limit(**search_limits), game=analysis_counter
                )
                fen = board.fen(en_passant="fen")
                score = info["score"].relative.score(mate_score=12800)
                best_move = info["pv"][0]
                file.write(f"{fen},{score},{best_move}\n")
                logging.info("Wrote result to file!")

            except chess.engine.EngineTerminatedError:
                if engine_generator is None:
                    logging.error("Can't restart engine due to missing generator")
                    raise

                # Try to kill the failed engine
                logging.info(f"Trying to kill engine")
                engine_generator.kill_engine(engine=engine)

                # Try to restart the engine
                logging.info("Trying to restart engine")

                engine = await engine_generator.get_initialized_engine(initialize_network=False)
            finally:
                queue.task_done()


async def analyze_positions_stockfish(
    engine_generator: EngineGenerator,
    data_generator: BoardGenerator,
    *,
    search_limits: Optional[Dict[str, Any]] = None,
    result_file_path: Optional[Union[str, Path]] = None,
    num_positions: int = 1,
    sleep_between_positions: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[chess.Board], List[Tuple[float, float]]]:

    if logger is None:
        logger = logging.getLogger(__name__)

    engine = await engine_generator.get_initialized_engine()

    # If no search limit has been specified, just search one single node
    if search_limits is None:
        search_limits = {"depth": 25}

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
            file_path=result_file_path,
            search_limits=search_limits,
            num_boards=num_positions,
            sleep_after_get=0.001,
            engine_generator=engine_generator,
        )
    )

    # Add callbacks to all tasks
    handle_task_exception = get_task_result_handler(
        logger=logger, message="Task raised an exception"
    )

    for task in [data_generator_task, data_consumer_task]:
        task.add_done_callback(handle_task_exception)

    # Wait until all tasks finish
    await asyncio.gather(data_generator_task, data_consumer_task)

    # await engine1.quit()
    # await engine2.quit()


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
        plot_board(
            board,
            title=title,
            fen=fen,
            show=show,
            save_path=file_path,
            fontsize=fontsize,
        )

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
    # parser.add_argument("--engine_config_name", type=str, default="remote_25_depth_stockfish.ini") # noqa: E501
    parser.add_argument("--data_config_name",   type=str, default="different_fen_database.ini")
    parser.add_argument("--num_positions",      type=int, default=16_950)
    parser.add_argument("--result_subdir",      type=str, default="stockfish_analysis")
    # fmt: on
    ##################################
    #           CONFIG END           #
    ##################################
    # Set up the logger
    logging.basicConfig(
        format="▸ %(asctime)s.%(msecs)03d %(filename)s:%(lineno)d %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()

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

    # Create results-file-name
    engine_config_name = args.engine_config_name[:-4]
    data_config_name = args.data_config_name[:-4]

    # Build the result file path
    result_directory = RESULT_DIR / args.result_subdir
    result_directory.mkdir(parents=True, exist_ok=True)
    result_file_path = result_directory / Path(
        f"results_ENGINE_{engine_config_name}_DATA_{data_config_name}.csv"
    )

    # Store the experiment configuration in the result file
    store_experiment_params(
        namespace=args, result_file_path=result_file_path, source_file_path=__file__
    )

    # Run the differential testing
    start_time = time.perf_counter()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(
        analyze_positions_stockfish(
            engine_generator=engine_generator,
            data_generator=data_generator,
            result_file_path=result_file_path,
            search_limits=engine_config.search_limits,
            num_positions=args.num_positions,
            logger=logger,
        )
    )

    end_time = time.perf_counter()
    logger.info(f"Elapsed time: {end_time - start_time: .3f} seconds")
