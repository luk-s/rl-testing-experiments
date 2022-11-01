import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import chess.engine
import numpy as np
from rl_testing.config_parsers import BoardGeneratorConfig, RemoteEngineConfig
from rl_testing.config_parsers.engine_config_parser import EngineConfig
from rl_testing.data_generators import (
    BoardGenerator,
    DataBaseBoardGenerator,
    FENDatabaseBoardGenerator,
    RandomBoardGenerator,
)
from rl_testing.engine_generators import EngineGenerator, RemoteEngineGenerator
from rl_testing.engine_generators.relaxed_uci_protocol import RelaxedUciProtocol
from rl_testing.util.util import MoveStat, PositionStat, parse_info

RESULT_DIR = Path(__file__).parent / Path("results/forced_moves")


async def analyze_positions(
    queue: asyncio.Queue,
    engine: RelaxedUciProtocol,
    search_limits: Dict[str, Any],
    num_boards: int,
    sleep_after_get: float = 0.0,
    engine_generator: Optional[EngineGenerator] = None,
    network_name: Optional[str] = None,
) -> List[Tuple[Union[chess.Move, str], Dict[chess.Move, MoveStat], List[PositionStat]]]:
    results = []
    # Iterate over all boards
    for board_index in range(num_boards):
        # Fetch the next board from the queue
        board = await queue.get()

        await asyncio.sleep(delay=sleep_after_get)

        print(f"Analyzing board {board_index + 1}/{num_boards}: " + board.fen(en_passant="fen"))

        move_stats, position_stats = {}, []

        # Needs to be in a try-except because the engine might crash unexpectedly
        try:
            # Start the analysis
            with await engine.analysis(
                board, limit=chess.engine.Limit(**search_limits)
            ) as analysis:
                async for info in analysis:
                    # Parse the analysis info
                    result = parse_info(info=info, raise_exception=False)

                    # Check if the info could be parsed into a 'MoveStat'
                    # or a 'PositionStat' instance
                    if result is not None:
                        if isinstance(result, MoveStat):
                            move_stats[result.move] = result
                        elif isinstance(result, PositionStat):
                            position_stats.append(result)
                        else:
                            ValueError(f"Objects of type {type(result)} are not supported")

        except chess.engine.EngineTerminatedError:
            if engine_generator is None or network_name is None:
                print("Can't restart engine due to missing generator")
                raise

            # Mark the current board as failed
            results.append(("invalid", {}, []))

            # Try to restart the engine
            print("Trying to restart engine")

            engine_generator.set_network(network_name)
            engine = await engine_generator.get_initialized_engine()

        else:
            # Check if the proposed best move is valid
            if engine.invalid_best_move:
                best_move = (await analysis.wait()).move
                results.append(("invalid", {}, []))
            else:
                best_move = (await analysis.wait()).move
                results.append((best_move, move_stats, position_stats))
        finally:
            queue.task_done()

    return results


async def get_positions_async(
    first_queue: asyncio.Queue,
    second_queue: asyncio.Queue,
    data_generator: BoardGenerator,
    num_positions: int = 1,
    sleep_between_positions: float = 0.1,
) -> List[Tuple[chess.Board, chess.Board]]:
    board_tuples = []

    # Create random chess positions if necessary
    for board_index in range(num_positions):

        # Create a random chess position
        board_candidate = data_generator.next()

        # Check if the generated position was valid
        if board_candidate != "failed" and len(list(board_candidate.legal_moves)) == 1:
            # Log the board position
            fen = board_candidate.fen(en_passant="fen")
            print(f"Created board {board_index}: " f"{fen}")

            # Push the board position to the analysis queue
            await first_queue.put(board_candidate.copy())

            # Find the only legal move
            move = list(board_candidate.legal_moves)[0]
            board2 = board_candidate.copy()
            board2.push(move)

            # Push the new board position to the analysis queue
            await second_queue.put(board2.copy())

            board_tuples.append((board_candidate, board2))

            await asyncio.sleep(delay=sleep_between_positions)

    return board_tuples


async def forced_moves_testing(
    network_name: str,
    engine_generator: EngineGenerator,
    data_generator: BoardGenerator,
    *,
    search_limits: Optional[Dict[str, Any]] = None,
    num_positions: int = 1,
    sleep_between_positions: float = 0.1,
) -> Tuple[List[chess.Board], List[Tuple[float, float]]]:

    # Create two instances of the same network
    engine_generator.set_network(network_name)
    engine1 = await engine_generator.get_initialized_engine()
    engine2 = await engine_generator.get_initialized_engine()

    # If no search limit has been specified, just search one single node
    if search_limits is None:
        search_limits = {"nodes": 1}

    results = []
    board_tuples_final = []
    queue1, queue2 = asyncio.Queue(), asyncio.Queue()

    data_generator_task = asyncio.create_task(
        get_positions_async(
            first_queue=queue1,
            second_queue=queue2,
            data_generator=data_generator,
            num_positions=num_positions,
            sleep_between_positions=sleep_between_positions,
        )
    )

    (data_consumer_task1, data_consumer_task2) = [
        asyncio.create_task(
            analyze_positions(
                queue=queue,
                engine=engine,
                search_limits=search_limits,
                num_boards=num_positions,
                sleep_after_get=sleep_time,
                engine_generator=engine_generator,
                network_name=network_name,
            )
        )
        for queue, engine, sleep_time in zip([queue1, queue2], [engine1, engine2], [0.05, 0.05])
    ]

    # Wait until all tasks finish
    board_tuples, results1, results2 = await asyncio.gather(
        data_generator_task, data_consumer_task1, data_consumer_task2
    )

    # await engine1.quit()
    # await engine2.quit()

    for board_index, (board1, board2) in enumerate(board_tuples):
        best_move1, move_stats1, _ = results1[board_index]
        best_move2, move_stats2, _ = results2[board_index]

        if best_move1 != "invalid" and best_move2 != "invalid":
            results.append((move_stats1[best_move1].Q, move_stats2[best_move2].Q))
            board_tuples_final.append((board1, board2))
        else:
            message = "board: " + board1.fen(en_passant="fen") + " "
            if best_move1 == "invalid":
                message += "engine 1 invalid, "
            if best_move2 == "invalid":
                message += "engine 2 invalid"
            print(message)

    return board_tuples_final, results


if __name__ == "__main__":
    ##################################
    #           CONFIG START         #
    ##################################
    SEED = 42
    ENGINE_CONFIG_NAME = "lukas_local_400_nodes.ini"  # "remote_400_nodes.ini"
    DATA_CONFIG_NAME = "forced_moves_fen_database.ini"  # "random_many_pieces.ini"
    REMOTE = False
    POSITIONS = []
    NUM_POSITIONS = 100
    # NETWORK_PATH = "network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42"
    # "network_c8368caaccd43323cc513465fb92740ea6d10b50684639a425fca2b42fc1f7be"
    NETWORK_PATH = "f21ee51844a7548c004a1689eacd8b4cd4c6150d6e03c732b211cf9963d076e1"
    # NETWORK_PATH = "fbd5e1c049d5a46c098f0f7f12e79e3fb82a7a6cd1c9d1d0894d0aae2865826f"

    RESULT_SUBDIR = "main_experiment"
    ##################################
    #           CONFIG END           #
    ##################################

    np.random.seed(SEED)

    # Build the engine generator
    if REMOTE:
        RemoteEngineConfig.set_config_folder_path(
            Path(__file__).parent.absolute() / Path("configs/engine_configs/")
        )
        engine_config = RemoteEngineConfig.from_config_file(ENGINE_CONFIG_NAME)
        engine_generator = RemoteEngineGenerator(engine_config)

    else:
        EngineConfig.set_config_folder_path(
            Path(__file__).parent.absolute() / Path("configs/engine_configs/")
        )
        engine_config = EngineConfig.from_config_file(ENGINE_CONFIG_NAME)
        engine_generator = EngineGenerator(engine_config)

    # Build the data generator
    BoardGeneratorConfig.set_config_folder_path(
        Path(__file__).parent.absolute() / Path("configs/data_generator_configs")
    )
    data_config = BoardGeneratorConfig.from_config_file(DATA_CONFIG_NAME)
    # data_generator = RandomBoardGenerator(**data_config.board_generator_config)
    data_generator = FENDatabaseBoardGenerator(**data_config.board_generator_config)

    # Run the differential testing
    start_time = time.perf_counter()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    board_tuples, results = asyncio.run(
        forced_moves_testing(
            network_name=NETWORK_PATH,
            data_generator=data_generator,
            engine_generator=engine_generator,
            search_limits=engine_config.search_limits,
            num_positions=NUM_POSITIONS,
        )
    )

    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time: .3f} seconds")

    # Create results-file-name
    engine_config_name = ENGINE_CONFIG_NAME[:-4]
    data_config_name = DATA_CONFIG_NAME[:-4]

    # Open the results file
    result_directory = RESULT_DIR / RESULT_SUBDIR
    result_directory.mkdir(parents=True, exist_ok=True)
    with open(
        result_directory
        / Path(f"results_ENGINE_{engine_config_name}_DATA_{data_config_name}.txt"),
        "a",
    ) as f:
        # Store the config
        f.write(f"{SEED = }\n")
        f.write(f"{ENGINE_CONFIG_NAME = }\n")
        f.write(f"{DATA_CONFIG_NAME = }\n")
        f.write(f"{REMOTE = }\n")
        f.write(f"{POSITIONS = }\n")
        f.write(f"{NUM_POSITIONS = }\n")
        f.write(f"{NETWORK_PATH = }\n")
        f.write(f"{RESULT_SUBDIR = }\n")
        f.write("\n")

        # Store the results
        f.write("FEN1, FEN2, Q1, Q2\n")
        for (board1, board2), (q1, q2) in zip(board_tuples, results):
            fen1 = board1.fen(en_passant="fen").replace(" ", "_")
            fen2 = board2.fen(en_passant="fen").replace(" ", "_")
            f.write(f"{fen1},{fen2},{q1},{q2}\n")
