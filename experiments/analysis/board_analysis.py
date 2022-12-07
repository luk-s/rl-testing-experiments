import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chess
import netwulf as nw
from chess.engine import Score

from rl_testing.config_parsers import get_engine_config
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.util.tree_parser import convert_tree_to_networkx


async def analyze_with_engine(
    engine_generator: EngineGenerator,
    positions: List[Union[chess.Board, str]],
    network_name: Optional[str] = None,
    search_limits: Optional[Dict[str, Any]] = None,
) -> List[Score]:
    engine_scores = []
    trees = []

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
        engine_scores.append(info["score"].relative)
        if "mcts_tree" in info:
            trees.append(info["mcts_tree"])

    return engine_scores, trees


if __name__ == "__main__":
    fens = ["r1bq3r/ppp1n1pp/1b6/4P3/3k4/6P1/P4P1P/R2QR1K1 b - - 4 21"]

    network_name = "network_d295bbe9cc2efa3591bbf0b525ded076d5ca0f9546f0505c88a759ace772ea42"
    # network_name = "network_c8368caaccd43323cc513465fb92740ea6d10b50684639a425fca2b42fc1f7be"
    # network_name = "network_600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2"

    # engine_config_name = "remote_25_depth_stockfish.ini"
    # search_limit = {"depth": 25}
    engine_config_name = "remote_debug_1000_nodes.ini"
    # engine_config_name = "remote_400_nodes.ini"
    search_limit = {"nodes": 1000}

    # Analyze the positions with stockfish
    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())

    # Setup engine generator
    engine_config = get_engine_config(
        config_name=engine_config_name,
        config_folder_path=Path(__file__).parent.parent.absolute()
        / Path("configs/engine_configs/"),
    )
    engine_generator = get_engine_generator(engine_config)

    scores, trees = asyncio.run(
        analyze_with_engine(
            engine_generator=engine_generator,
            positions=fens,
            network_name=network_name,
            search_limits=search_limit,
        )
    )

    print("Results:")
    for fen, score, tree in zip(fens, scores, trees):
        print(f"board {fen :<74} score: {score}")
        graph = convert_tree_to_networkx(tree, only_basic_info=True)
        stylized_network, config = nw.visualize(graph)
