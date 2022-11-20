import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chess
from chess.engine import Score

from rl_testing.config_parsers import get_engine_config
from rl_testing.engine_generators import EngineGenerator, get_engine_generator


async def analyze_with_engine(
    engine_generator: EngineGenerator,
    positions: List[Union[chess.Board, str]],
    search_limits: Optional[Dict[str, Any]] = None,
) -> List[Score]:
    engine_scores = []

    # Set search limits
    if search_limits is None:
        search_limits = {"depth": 25}

    # Setup and configure the engine
    # TODO: Remove after test!
    engine_generator.set_network(
        "network_c8368caaccd43323cc513465fb92740ea6d10b50684639a425fca2b42fc1f7be"
    )
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

    return engine_scores


if __name__ == "__main__":
    fens = [
        "1qk4r/pp5r/4b1p1/nNp1Np2/4P3/6QP/PPP3BK/3R4 b - - 3 32",
        "4rr2/2q2p1k/p6p/1p1bp2P/8/P1PPQN2/3K1PR1/2R5 b - - 3 33",
        "2r2kn1/1br2pp1/p2q1b1p/1p2pP1Q/3pP1NP/1B1P2P1/PP5N/5RRK w - - 9 32",
        "k1brr3/ppq3pp/1Rp4n/Q1P5/R3p3/3B1N1P/P4PP1/6K1 b - - 1 26",
        "1qk4r/pp5r/4b1p1/n1p1Np2/4P3/2N3QP/PPP3BK/3R4 w - - 2 32",
        "6rk/1bp1qp1p/p6Q/1p1ppPp1/3bP2P/2NP3R/PPP3r1/1K1R4 w - - 0 23",
        "8/7p/2k1p1p1/1n6/p1rP4/5P1P/3K1BP1/1R6 w - - 6 57",
        "r1bq3r/ppp1n1pp/1b6/2k1P3/8/6P1/P4P1P/R2QR1K1 w - - 5 22",
        "4rn1k/p1pb2p1/2pp1p1p/4qP2/4PN2/P2B2QP/1rP3P1/1R3R1K w - - 0 26",
        "3r2k1/5p1p/p3q1pP/6P1/1pR2B2/1Pb1PNK1/P1Q2P2/3r4 w - - 6 40",
        "r1bq3r/ppp1n1pp/1b6/4P3/3k4/6P1/P4P1P/R2QR1K1 b - - 4 21",
        "8/1R6/8/8/8/r3KP2/2p5/2k5 w - - 20 85",
        "k1brr3/ppq3pp/1Rp4n/2P1p3/R7/3B1N1P/P2Q1PP1/6K1 b - - 1 25",
        "1r2rn1k/p1pb2p1/2pp1p1p/4qP2/4PN2/P2B2QP/1PP3P1/1R3R1K b - - 3 25",
        "6rk/1bp1q2p/p4p1Q/1p1ppPP1/3bP3/2NP3R/PPP3r1/1K1R4 w - - 0 24",
        "6rk/1bp1qp1p/p6Q/1p1ppPP1/3bP3/2NP3R/PPP3r1/1K1R4 b - - 0 23",
        "k1brr3/ppq3pp/1Rp4n/2P5/R3p3/3B1N1P/P2Q1PP1/6K1 w - - 0 26",
        "r4k2/1pp1brqp/p2pR3/3P4/1P3P2/4Q3/P1P3PP/4R1K1 w - - 1 25",
        "1k6/8/P1K5/2PR4/8/8/2r5/8 b - - 0 80",
    ]
    # engine_config_name = "remote_25_depth_stockfish.ini"
    # search_limit = {"depth": 25}
    engine_config_name = "remote_1000_nodes.ini"
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

    scores = asyncio.run(
        analyze_with_engine(
            engine_generator=engine_generator,
            positions=fens,
            search_limits=search_limit,
        )
    )

    print("Results:")
    for fen, score in zip(fens, scores):
        print(f"board {fen :<74} score: {score}")
