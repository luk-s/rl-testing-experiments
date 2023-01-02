import argparse
import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import chess
import chess.engine
import numpy as np

from rl_testing.config_parsers import get_data_generator_config, get_engine_config
from rl_testing.data_generators import BoardGenerator, get_data_generator
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.engine_generators.relaxed_uci_protocol import RelaxedUciProtocol
from rl_testing.util.util import (
    get_task_result_handler,
    cp2q,
)
from rl_testing.util.experiment import store_experiment_params

RESULT_DIR = Path(__file__).parent / Path("results/forced_moves")


async def get_positions(
    original_board_queue: asyncio.Queue,
    forced_move_board_queue: asyncio.Queue,
    data_generator: BoardGenerator,
    num_positions: int = 1,
    sleep_between_positions: float = 0.1,
    identifier_str: str = "",
) -> None:
    num_boards_created = 0

    while num_boards_created < num_positions:
        # Get the next board position
        board_candidate = data_generator.next()

        # Check if the generated position was valid and contains a forced move
        if board_candidate == "failed":
            continue
        if len(list(board_candidate.legal_moves)) != 1:
            continue

        # Find the only legal move
        move = list(board_candidate.legal_moves)[0]
        board2 = board_candidate.copy()
        board2.push(move)

        # Only consider positions where you can make predictions
        if len(list(board2.legal_moves)) == 0:
            await original_board_queue.put("failed")
            await forced_move_board_queue.put("failed")
            continue

        # Log the board position
        fen = board_candidate.fen(en_passant="fen")
        logging.info(
            f"[{identifier_str}] Created board {num_boards_created + 1}/{num_positions}: "
            f"{fen} with forced move {move}"
        )

        # Push the board position to the analysis queue
        await original_board_queue.put(board_candidate.copy())

        # Push the new board position to the analysis queue
        await forced_move_board_queue.put(board2.copy())

        await asyncio.sleep(delay=sleep_between_positions)

        num_boards_created += 1


async def analyze_positions(
    input_queue: asyncio.Queue,
    output_queue: asyncio.Queue,
    engine: RelaxedUciProtocol,
    search_limits: Dict[str, Any],
    sleep_after_get: float = 0.0,
    engine_generator: Optional[EngineGenerator] = None,
    network_name: Optional[str] = None,
    identifier_str: str = "",
) -> None:
    num_boards_analyzed = 0

    # Iterate over all boards
    while True:
        # Fetch the next board from the queue
        board = await input_queue.get()

        await asyncio.sleep(delay=sleep_after_get)

        if board == "failed":
            continue

        fen = board.fen(en_passant="fen")

        logging.info(
            f"[{identifier_str}] Analyzing board {num_boards_analyzed + 1}: "
            + board.fen(en_passant="fen")
        )

        # Needs to be in a try-except because the engine might crash unexpectedly
        try:
            info = await engine.analyse(board, chess.engine.Limit(**search_limits))
            best_move = info["pv"][0]
            score = cp2q(info["score"].relative.score(mate_score=12800))
        except chess.engine.EngineTerminatedError:
            if engine_generator is None or network_name is None:
                logging.info(
                    f"[{identifier_str}] Can't restart engine due to missing generator"
                )
                raise

            # Mark the current board as failed
            await output_queue.put((fen, "invalid", "invalid"))

            # Try to restart the engine
            logging.info(f"[{identifier_str}] Trying to restart engine")

            engine_generator.set_network(network_name)
            engine = await engine_generator.get_initialized_engine()

        else:
            # Check if the proposed best move is valid
            if engine.invalid_best_move:
                await output_queue.put((fen, "invalid", "invalid"))
            else:
                await output_queue.put((fen, best_move, score))
        finally:
            input_queue.task_done()

        num_boards_analyzed += 1


async def analyze_results(
    original_result_queue: asyncio.Queue,
    forced_move_result_queue: asyncio.Queue,
    file_path: Union[str, Path],
    sleep_after_get: float = 0.1,
    identifier_str: str = "",
) -> None:

    with open(file_path, "a") as file:
        # Write the file header
        file.write(
            "fen_original,best_move_original,score_original,"
            "fen_forced,best_move_forced,score_forced\n"
        )

        # Repeatedly fetch results from the queues
        while True:
            (
                fen_original,
                best_move_original,
                score_original,
            ) = await original_result_queue.get()
            (
                fen_forced,
                best_move_forced,
                score_forced,
            ) = await forced_move_result_queue.get()
            await asyncio.sleep(delay=sleep_after_get)

            logging.info(f"[{identifier_str}] Received results for {fen_original}")

            # Write the results to the file
            file.write(
                f"{fen_original},{best_move_original},{score_original},"
                f"{fen_forced},{best_move_forced},{score_forced}\n"
            )


async def forced_moves_testing(
    network_name: str,
    engine_generator: EngineGenerator,
    data_generator: BoardGenerator,
    result_file_path: Union[str, Path],
    *,
    search_limits: Optional[Dict[str, Any]] = None,
    num_positions: int = 1,
    sleep_between_positions: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> None:

    # Set up the logger
    if logger is None:
        logger = logging.getLogger(__name__)

    # Create two instances of the same network
    engine_generator.set_network(network_name)
    engine1 = await engine_generator.get_initialized_engine()
    engine2 = await engine_generator.get_initialized_engine()

    # If no search limit has been specified, just search one single node
    if search_limits is None:
        search_limits = {"nodes": 1}

    results = []
    board_tuples_final = []
    original_board_queue, forced_move_board_queue = asyncio.Queue(), asyncio.Queue()
    original_result_queue, forced_move_result_queue = asyncio.Queue(), asyncio.Queue()

    # Create the data generator task
    data_generator_task = asyncio.create_task(
        get_positions(
            original_board_queue=original_board_queue,
            forced_move_board_queue=forced_move_board_queue,
            data_generator=data_generator,
            num_positions=num_positions,
            sleep_between_positions=sleep_between_positions,
            identifier_str="BOARD GENERATOR",
        )
    )

    # Create the analysis tasks
    (data_consumer_task1, data_consumer_task2) = [
        asyncio.create_task(
            analyze_positions(
                input_queue=queue1,
                output_queue=queue2,
                engine=engine,
                search_limits=search_limits,
                sleep_after_get=sleep_time,
                engine_generator=engine_generator,
                network_name=network_name,
                identifier_str=identifier_str,
            )
        )
        for queue1, queue2, engine, sleep_time, identifier_str in zip(
            [original_board_queue, forced_move_board_queue],
            [original_result_queue, forced_move_result_queue],
            [engine1, engine2],
            [0.05, 0.05],
            ["ANALYSIS 1", "ANALYSIS 2"],
        )
    ]

    # Create the analysis task
    analysis_task = asyncio.create_task(
        analyze_results(
            original_result_queue=original_result_queue,
            forced_move_result_queue=forced_move_result_queue,
            file_path=result_file_path,
            sleep_after_get=0.1,
            identifier_str="RESULT",
        )
    )

    # Add callbacks to all tasks
    handle_task_exception = get_task_result_handler(
        logger=logger, message="Task raised an exception"
    )
    for task in [
        data_generator_task,
        data_consumer_task1,
        data_consumer_task2,
        analysis_task,
    ]:
        task.add_done_callback(handle_task_exception)

    # Wait for the data generator task to finish
    await asyncio.wait([data_generator_task])

    # Join the queues
    await original_board_queue.join()
    await forced_move_board_queue.join()
    await original_result_queue.join()
    await forced_move_result_queue.join()

    # Cancel all remaining tasks
    for task in [
        data_consumer_task1,
        data_consumer_task2,
        analysis_task,
    ]:
        task.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ##################################
    #           CONFIG START         #
    ##################################

    # NETWORKS:
    # =========
    # strong and recent: "network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42"
    # from leela paper: "network_c8368caaccd43323cc513465fb92740ea6d10b50684639a425fca2b42fc1f7be"
    # NETWORK_PATH = "f21ee51844a7548c004a1689eacd8b4cd4c6150d6e03c732b211cf9963d076e1"
    # NETWORK_PATH = "fbd5e1c049d5a46c098f0f7f12e79e3fb82a7a6cd1c9d1d0894d0aae2865826f"

    # fmt: off
    parser.add_argument("--seed",               type=int, default=42)
    # parser.add_argument("--engine_config_name", type=str, default="remote_400_nodes.ini")
    parser.add_argument("--engine_config_name", type=str, default="local_400_nodes.ini")
    parser.add_argument("--data_config_name",   type=str, default="late_move_fen_database.ini")
    parser.add_argument("--num_positions",      type=int, default=100_000)
    parser.add_argument("--network_path",       type=str, default="network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42")  # noqa: E501
    parser.add_argument("--result_subdir",      type=str, default="")
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

    # Parse the arguments
    args = parser.parse_args()

    np.random.seed(args.seed)

    engine_config = get_engine_config(
        config_name=args.engine_config_name,
        config_folder_path=Path(__file__).parent.absolute()
        / Path("configs/engine_configs/"),
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

    # Open the results file
    result_directory = RESULT_DIR / args.result_subdir
    result_directory.mkdir(parents=True, exist_ok=True)
    result_file_path = result_directory / Path(
        f"results_ENGINE_{engine_config_name}_DATA_{data_config_name}.txt"
    )
    store_experiment_params(
        namespace=args, result_file_path=result_file_path, source_file_path=__file__
    )

    # Run the differential testing
    start_time = time.perf_counter()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(
        forced_moves_testing(
            network_name=args.network_path,
            data_generator=data_generator,
            engine_generator=engine_generator,
            result_file_path=result_file_path,
            search_limits=engine_config.search_limits,
            num_positions=args.num_positions,
            logger=logger,
        )
    )

    end_time = time.perf_counter()
    logging.info(f"Elapsed time: {end_time - start_time: .3f} seconds")
