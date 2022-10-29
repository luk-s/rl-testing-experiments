import asyncio
import pathlib
import time
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncssh
import chess
import chess.engine
import numpy as np
from rl_testing.config_parsers.engine_config_parser import RemoteEngineConfig
from rl_testing.util.relaxed_uci_protocol import RelaxedUciProtocol
from rl_testing.util.util import (
    CHESS_PIECES_NON_ESSENTIAL,
    MoveStat,
    PositionStat,
    parse_info,
    plot_board,
    random_valid_board,
)


async def analyze_positions(
    # engine: chess.engine.UciProtocol, board: chess.Board
    engine: RelaxedUciProtocol,
    boards: List[chess.Board],
    search_limits: Dict[str, Any],
) -> List[Tuple[Union[chess.Move, str], Dict[chess.Move, MoveStat], List[PositionStat]]]:
    results = []
    # Iterate over all boards
    for board_index, board in enumerate(boards):
        print(f"Analyzing board {board_index}/{len(boards)}: " + board.fen(en_passant="fen"))

        move_stats, position_stats = {}, []

        # Start the analysis
        with await engine.analysis(board, limit=chess.engine.Limit(**search_limits)) as analysis:
            async for info in analysis:
                # Parse the analysis info
                result = parse_info(info=info, raise_exception=False)

                # Check if the info could be parsed into a 'MoveStat' or a 'PositionStat' instance
                if result is not None:
                    if isinstance(result, MoveStat):
                        move_stats[result.move] = result
                    elif isinstance(result, PositionStat):
                        position_stats.append(result)
                    else:
                        ValueError(f"Objects of type {type(result)} are not supported")

        # Check if the proposed best move is valid
        if engine.invalid_best_move:
            results.append(("invalid", {}, []))
        else:
            best_move = (await analysis.wait()).move
            results.append((best_move, move_stats, position_stats))

    return results


async def differential_testing(
    remote_host: str,
    remote_user: str,
    password_required: bool,
    engine_path: str,
    engine_config: str,
    network_path1: str,
    network_path2: str,
    positions: List[chess.Board] = [],
    num_positions: int = 1,
    search_limits: Optional[Dict[str, Any]] = None,
    num_pieces: Optional[int] = None,
    num_pieces_min: Optional[int] = None,
    num_pieces_max: Optional[int] = None,
    max_attempts_per_position: int = 10000,
):
    remote_password = None

    if remote_password is None and password_required:
        remote_password = getpass(
            prompt=f"Please specify the SSH password for the user {remote_user}:\n"
        )

    if positions:
        boards = positions
    else:
        boards = []

    # Start connection
    async with asyncssh.connect(
        remote_host, username=remote_user, password=remote_password
    ) as conn:
        del remote_password

        if not boards:
            # Create random chess positions if necessary
            for board_index in range(num_positions):
                # Choose how many pieces the position should have
                if num_pieces is None:
                    if num_pieces_min is None:
                        pieces_min = 1
                    else:
                        pieces_min = num_pieces_min
                    if num_pieces_max is None:
                        pieces_max = len(CHESS_PIECES_NON_ESSENTIAL)
                    else:
                        pieces_max = num_pieces_max
                    num_pieces_to_choose = np.random.randint(pieces_min, pieces_max)
                else:
                    num_pieces_to_choose = num_pieces

                # Create a random chess position
                board_candidate = random_valid_board(
                    num_pieces=num_pieces_to_choose,
                    max_attempts_per_position=max_attempts_per_position,
                )

                # Check if the generated position was valid
                if board_candidate != "failed":
                    boards.append(board_candidate)
                    fen = board_candidate.fen(en_passant="fen")
                    print(f"Created board {board_index}: " f"{fen}")

        # Connect the two networks to test
        _, engine1 = await conn.create_subprocess(
            # chess.engine.UciProtocol,
            RelaxedUciProtocol,
            engine_path,
        )

        _, engine2 = await conn.create_subprocess(
            # chess.engine.UciProtocol,
            RelaxedUciProtocol,
            engine_path,
        )

        # Initialize the two engines
        await engine1.initialize()
        await engine2.initialize()

        # Configur engine 1
        engine1_config = dict(engine_config)
        engine1_config["WeightsFile"] = network_path1
        await engine1.configure(engine1_config)

        # Configure engine 2
        engine2_config = dict(engine_config)
        engine2_config["WeightsFile"] = network_path2
        await engine2.configure(engine2_config)

        # If no search limit has been specified, just search one single node
        if search_limits is None:
            search_limits = {"nodes": 1}

        results = []
        boards_final = []

        results1, results2 = await asyncio.gather(
            analyze_positions(engine=engine1, boards=boards, search_limits=search_limits),
            analyze_positions(engine=engine2, boards=boards, search_limits=search_limits),
        )
        for board_index, board in enumerate(boards):
            best_move1, move_stats1, position_stats1 = results1[board_index]
            best_move2, move_stats2, position_stats2 = results2[board_index]

            if best_move1 != "invalid" and best_move2 != "invalid":
                results.append((move_stats1[best_move1].Q, move_stats2[best_move2].Q))
                boards_final.append(board)

    return boards_final, results


def analyze_results(
    boards: List[chess.Board],
    results: List[Tuple[float, float]],
    difference_threshold: float = 1.0,
    show: bool = True,
) -> None:
    # Find results where the models have a large difference
    results = np.array(results)
    differences = np.abs(results[:, 0] - results[:, 1])
    indices = np.where(differences >= difference_threshold)[0]

    special_results = []

    for index in indices:
        board = boards[index]
        value1, value2 = results[index]
        special_results.append((board, differences[index], value1, value2))
        if show:
            title = f"value1: {value1}, value2: {value2}"
            plot_board(board, title=title, fen=board.fen(en_passant="fen"))

    return special_results


if __name__ == "__main__":
    ##################################
    #           CONFIG START         #
    ##################################
    SEED = 42
    USE_DEFAULT_CONFIG = False
    ENGINE_CONFIG_NAME = "remote_400_nodes.ini"
    POSITIONS = []
    NUM_POSITIONS = 100
    NUM_PIECES = None
    NUM_PIECES_MIN = 10  # 18
    NUM_PIECES_MAX = None
    DIFFERENCE_THRESHOLD = 1
    NETWORK_PATH1 = "network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42"
    NETWORK_PATH2 = "network_c8368caaccd43323cc513465fb92740ea6d10b50684639a425fca2b42fc1f7be"
    ##################################
    #           CONFIG END           #
    ##################################

    np.random.seed(SEED)
    RemoteEngineConfig.set_config_folder_path(
        Path(__file__).parent.absolute() / Path("configs/engine_configs/")
    )
    if USE_DEFAULT_CONFIG:
        config = RemoteEngineConfig.default_config()
    else:
        config = RemoteEngineConfig.from_config_file(ENGINE_CONFIG_NAME)

    start_time = time.perf_counter()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    boards, results = asyncio.run(
        differential_testing(
            remote_host=config.remote_host,
            remote_user=config.remote_user,
            password_required=config.password_required,
            engine_path=config.engine_path,
            engine_config=config.engine_config,
            network_path1=str(Path(config.network_base_path) / Path(NETWORK_PATH1)),
            network_path2=str(Path(config.network_base_path) / Path(NETWORK_PATH2)),
            positions=POSITIONS,
            search_limits=config.search_limits,
            num_positions=NUM_POSITIONS,
            num_pieces=NUM_PIECES,
            num_pieces_min=NUM_PIECES_MIN,
            num_pieces_max=NUM_PIECES_MAX,
        )
    )
    print("Results = ", results)

    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time: .3f} seconds")
    if results:
        special_cases = analyze_results(
            boards, results, difference_threshold=DIFFERENCE_THRESHOLD, show=True
        )
        print(
            f"Found {len(special_cases)} boards where "
            f"evaluation differs by more than {DIFFERENCE_THRESHOLD}!"
        )
        for board, difference, v1, v2 in special_cases:
            fen = board.fen(en_passant="fen")
            print(f"{fen}, {difference = :.2f}, {v1 = :.3f}, {v2 = :.3f}")
