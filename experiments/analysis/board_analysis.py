import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import netwulf as nw
from chess.engine import Score

from rl_testing.config_parsers import get_engine_config
from rl_testing.engine_generators import EngineGenerator, get_engine_generator
from rl_testing.mcts.tree_parser import TreeInfo, convert_tree_to_networkx
from rl_testing.util.util import cp2q


async def analyze_with_engine(
    engine_generator: EngineGenerator,
    positions: List[Union[chess.Board, str]],
    network_name: Optional[str] = None,
    search_limits: Optional[Dict[str, Any]] = None,
    score_type: str = "cp",
) -> Tuple[List[Score], List[TreeInfo]]:
    valid_score_types = ["cp", "q"]
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
        cp_score = info["score"].relative.score(mate_score=12800)
        if score_type == "cp":
            engine_scores.append(cp_score)
        elif score_type == "q":
            engine_scores.append(cp2q(cp_score))
        else:
            raise ValueError(
                f"Invalid score type: {score_type}. Choose one from {valid_score_types}"
            )

        if "mcts_tree" in info:
            trees.append(info["mcts_tree"])

    return engine_scores, trees


if __name__ == "__main__":
    ################
    # CONFIG START #
    ################
    fens = [
        "r4n1k/p6p/2p1q1r1/1p1n4/3P1PPQ/2P4R/P2B3P/R5K1 w - - 0 25",
        "8/1p3n2/1p6/pPp1p3/P1P1P1k1/1K1P4/3B4/8 w - - 112 119",
        "8/1p6/1p6/pPp1p1n1/P1P1P1k1/1K1P4/3B4/8 b - - 111 118",
        "8/1p6/1p6/pPp1p1n1/P1P1P1k1/1K1P4/8/2B5 w - - 110 118",
        "r3k2r/pbpq1ppp/2Nb4/8/Q3p3/2P1P3/P2P1PPP/R1B1K2R w KQkq - 3 12",
        "5rk1/5p1p/4n1pB/3QP3/3P4/8/qpp2PPP/5RK1 w - - 0 33",
        "2k5/1p6/pBp5/3b4/2P3n1/1P4P1/P3K2p/3R4 b - c3 0 41",
        "rnb2rk1/pp1n2pp/4p3/3pP3/1q1P4/3B4/PP1QN1PP/R3K1NR b KQ - 2 13",
        "5q1k/pp1br1rp/2n2p2/2bQ4/5N1P/5NR1/PPP2PP1/3R1K2 w - - 2 23",
        "r2q1rk1/3p1ppp/b1p2n2/p7/PpB1P3/3Q4/1PP1NPPb/R1B2RK1 w - - 0 15",
        "Q7/ppp2k1p/3p2p1/5b2/4P1n1/2P4q/PP1P1bP1/RNB2R1K w - - 0 14",
        "rnb1k2r/ppp2ppp/3b4/3pq1n1/8/1B1P2P1/PPP2P1P/RNBQR1K1 b kq - 2 9",
        "8/5PK1/8/6k1/6p1/3p4/7P/8 b - - 0 68",
        "8/6K1/5P2/6k1/6p1/3p4/7P/8 w - - 0 68",
        "r4rk1/1p3ppp/1ppp4/4p3/3PPn2/1BN2Pn1/PPPQ2RN/3R2K1 b - - 0 21",
        "2r1nr1k/pp1q1p1p/3bpp2/5P2/1P1Q4/P3P3/1BP2P1P/R3K1R1 w Q - 2 18",
        "R1R5/p4pk1/1p2r3/4PK1p/5P2/P3r3/6p1/8 b - - 1 52",
        "4q1k1/5rb1/3p2r1/p1p4R/Pp2P3/3P1p1Q/1BP2P1P/7K w - - 1 39",
        "r1bqk2r/ppp2ppp/2n5/3p4/2BPn3/B1P2N2/P4PPP/R2Q1RK1 b kq - 1 10",
        "8/ppn2kp1/2p2pp1/3p4/3P1P2/2PQrNP1/PP4K1/5R2 b - - 0 33",
        "r3r1k1/pq3p2/4bR2/3p2Q1/3P2pP/2N3P1/P1P4K/8 b - - 2 24",
        "1br1r1k1/pp3pp1/7p/8/2n2P2/PQ2PR2/4N1Pq/R1B2K2 w - - 0 23",
        "8/1p6/p1p1P3/4RbQp/P4p1k/1P6/3r2PP/6K1 b - - 0 53",
        "6k1/8/p7/2p1P1rp/3p3P/8/PP4P1/6K1 w - - 0 38",
        "6k1/6p1/1P4qp/3b4/8/4QPK1/7P/8 w - - 12 61",
        "8/8/5k2/p1p1p3/P3P3/3P4/8/6K1 b - - 9 51",
        "R7/8/P5k1/r5p1/8/5PK1/8/8 w - - 3 63",
        "2r1nrk1/pp1q1ppp/3bpN2/5P2/1P1Q4/P3P3/1BP2P1P/R3K2R b KQ - 2 16",
        "3k4/1ppb1rp1/p2p3p/3Pr3/1PP1PK2/4R2P/P1B3P1/4R3 w - - 1 32",
        "8/8/6k1/5p1p/6p1/8/R5K1/8 w - - 0 53",
        "8/5p2/8/3P4/P4kpp/1P6/8/6K1 b - - 0 66",
        "r1bq4/ppppk3/2n1prB1/4b3/7Q/B7/P4PPP/1N3RK1 b - - 4 19",
        "2k5/4Q3/6q1/1p1p1p2/2pP1P2/2P5/r7/K2R4 w - - 0 48",
        "4rrk1/p1pqb1p1/2p2pP1/1p3n1b/3PN2R/2P1B2Q/PP4PP/R5K1 b - - 6 23",
        "rnb1r1k1/2q1bp1p/1p2pn2/p1p3p1/Q2P1BP1/P2BPNN1/1P1q1PKP/3R3R w - - 2 18",
        "r2q1b1r/pppb1Bpp/2np1k2/5pN1/4P3/2N5/PPP2PPP/R1B1K2R w KQ - 0 10",
        "r1bq1bnr/pppp2kp/8/7Q/4Pn2/2N5/PPP3PP/R4RK1 w - - 0 12",
        "5k2/8/1p1p3Q/1P1Pp3/4P1np/3r2q1/8/7K b - - 1 51",
        "rnbq2rk/pp2bR1p/2p4Q/3nP1B1/3P4/8/PPP3PP/RN4K1 b - - 8 15",
        "r6r/pbppk3/1p2pqQ1/6B1/2P5/8/PP2nPPP/R4R1K b - - 3 18",
        "2r1r3/p5Rk/3BQn1p/7q/8/8/P4PPP/1R4K1 b - - 0 34",
        "8/pbp3pk/2r5/3RPp1p/4qP2/1P2Q1P1/P4K1P/3R4 b - - 3 35",
        "r4r2/p3n1pk/2p1Q2p/n7/1bPq4/1P3b2/PB3P1P/R4KNR w - - 2 27",
        "r5k1/ppp2ppp/8/3P4/1q1n4/3n4/PPPQ2PP/R2BK2R w KQ - 0 23",
        "8/8/3p4/5k1p/1p1p4/1P1P2P1/5K2/8 b - - 3 54",
        "1r2qb2/1bp2p1k/3prB1p/2p2N1Q/2P1PP2/1P4R1/6PP/4R1K1 b - - 0 30",
        "r4n1k/p6p/2p1q1r1/1p1n1P2/3P2PQ/2P4R/P2B3P/R5K1 b - - 0 25",
        "1k6/2p5/1p1p2p1/p2Bn1P1/Q3b3/PPP4P/1KPR4/4q3 b - - 2 36",
        "5r2/p1p3kp/b1p2pp1/4P3/2q2n1Q/1R3N2/PR3PPP/7K b - - 0 26",
        "4q2k/5rb1/3p2r1/p1p4p/Pp2P2R/3P1p1Q/1BP2P1P/7K w - - 0 38",
        "1rb3k1/p1p3pp/2pb4/3p4/3Pp3/4B2q/PPPQBr1P/2KR3R w - - 0 18",
        "8/1p2r1k1/3PPp1r/2p1Q1p1/8/5R1P/6PK/8 b - - 0 39",
        "2rq2k1/5ppp/3p4/p1pN4/2QnP3/1P3P2/P5PP/3R3K w - - 1 33",
        "r3r1k1/pp1bb1pp/5p2/3R2B1/8/2NP3P/PPP2PP1/5RK1 b - - 0 21",
        "4n1k1/1p3pp1/7p/8/1R6/4Q2P/5PPK/3q4 b - - 2 35",
        "3R1nk1/5rp1/4Q3/4B3/3p1P2/8/6P1/4q1K1 w - - 10 42",
        "8/8/7K/7P/R7/6k1/4p3/6r1 w - - 9 95",
        "rnbk1b1r/ppBpqp1p/5N2/8/7P/3P4/PPP1Q1P1/RN2KB1n b Q - 0 11",
        "8/8/1pk5/p4R2/P1P5/1P4r1/6p1/6K1 b - - 13 64",
        "r6r/pp1qb1k1/2p5/3p1b1Q/3P1P2/1B4Rp/PPP3P1/R5K1 b - - 1 26",
        "1r5k/p3Q2r/b2p4/2p1pP2/3nP3/2PP4/q3BK2/4R3 w - - 0 34",
        "6R1/1r1k2p1/7p/3pp3/8/4K3/6PP/8 w - - 0 41",
        "6k1/R4pp1/4N3/4P1P1/8/4K3/Prb5/8 b - - 0 53",
        "7k/1pq2P2/p1p3Q1/2P4p/4p2p/2P5/P4PPK/3r4 w - - 0 44",
        "r5rk/ppp1q1b1/2npb2p/5N2/3Pp3/2P2N2/PPB1QPPP/4RR1K b - - 0 19",
        "2r3k1/2r2pn1/1pp1qQp1/p2pP1P1/P2P3P/1PR5/5B2/3R2K1 b - - 6 32",
        "2kr3r/pp4b1/2BP2p1/2P1P2p/6np/2P1P1P1/P7/1R3RNK b - - 0 28",
        "r2r2k1/1b4bp/pq4p1/2Nn1p2/3p1N2/1B1P4/1PPQ1PPP/R4RK1 b - - 0 20",
        "r1bq3r/ppp3pp/8/3nk3/2B5/2p2Q2/PPP2PPP/R5K1 w - - 0 15",
        "r2qkb1r/1pp2Bpp/p2p4/4N3/4P3/2N5/PPPP4/R1Bb1RK1 b kq - 0 12",
        "8/p4B2/1p1p1kPP/1r6/3P4/5K2/8/8 w - - 0 54",
        "8/p4p2/3k4/6P1/2pP4/1PpnK3/P7/5R2 b - - 1 36",
        "2b5/2p1Qr1k/2n3pn/p1N1p3/1pP1Pp1P/1P2BP2/P1B2K2/3R4 b - - 0 36",
        "k5r1/pp1b4/3P4/1P3rPp/3Q4/4Pp1P/P4B1K/8 b - - 6 35",
        "3k4/1P6/3P4/3K4/8/1r6/2p5/2R5 w - - 3 73",
        "r1b2rk1/p1pp2pp/3b3q/1p1N4/4Q3/1B1N2P1/PP3P1P/R5K1 b - - 0 21",
        "3q4/1b2r1kp/p2b1np1/3p1p2/2R5/P2QPB1P/1P3PP1/3R1NK1 b - - 0 27",
        "2k5/R7/6q1/1p1p1p2/2pP1P2/2P5/PP4rr/1K1RQ3 w - - 3 45",
        "4r1k1/p4pp1/1b1r1q1p/2pBp3/2P1P1b1/4QN2/PB1R1PnP/R5K1 w - - 0 24",
        "r3qRkr/ppp5/3p2pQ/8/3PP1b1/8/PPP3P1/5RK1 b - - 1 20",
        "2rq4/1b2r1kp/p2b1np1/3p1p2/2N5/P2QPB1P/1P3PP1/2RR1NK1 b - - 0 26",
        "r3k3/2pq1p2/p6Q/1p6/3Nr3/P6P/1P3PPK/R7 b - - 0 30",
        "6k1/1P6/8/2N2ppP/8/1b6/7r/5K2 b - - 0 55",
        "6k1/p1b2pp1/7p/8/3q2Q1/3b1PNP/6P1/6K1 w - - 0 40",
        "r2q1rk1/3p1ppp/b1p2n2/p7/PpB1P3/3Q4/1PP1NPPK/R1B2R2 b - - 0 15",
        "5r1k/p1p3p1/6p1/3Q4/4n3/2P1b3/PP3rPP/RN2R1K1 w - - 0 23",
        "6k1/pp4p1/2p5/2bp3r/8/4N1Pb/PP3r1P/2BRR2K w - - 24 39",
        "8/2R2pkp/1P2P1p1/3b2r1/r7/p1N1p2K/7P/4R3 w - - 0 35",
        "8/2Np4/1Pb5/2K5/8/7k/7P/8 w - - 1 49",
        "5rk1/1pp2pp1/6Np/p1N2n1q/P1QP2P1/5R1n/1P3PKP/3R4 b - - 0 24",
        "r4rk1/p3n1p1/2p1Q2p/n7/1bPq4/1P3b2/PB3P1P/R4KNR b - - 1 26",
        "5kn1/8/5p2/4pP2/1P1pP3/3P4/pr1N4/3r1Q1K w - - 0 56",
        "5rk1/1pR2p2/p1b1r1p1/3qB3/5P2/6RQ/P5PK/8 b - - 1 31",
        "r1b1k2r/ppp1nppp/2n5/1N1pq3/4P3/2P1Q3/PP3PPP/RN2KB1R w KQkq - 1 10",
        "rk5r/pppNbq1p/2n5/6B1/3p2Q1/8/PPP2PPP/R5K1 b - - 7 21",
        "1r6/p1k2p2/2p1pb2/7p/2N2p2/P1P5/1P1R1PPP/6K1 b - - 0 27",
        "4R3/8/3pkp2/2p3p1/P3r3/7P/6PK/8 b - - 1 49",
        "r2k1n2/ppp3Qp/2n1b3/3q4/8/3P4/PPP3PP/4R2K w - - 0 23",
        "2r2rk1/pp1q1ppp/3bpn2/3N1P2/1P1Q4/P3P3/1BP2P1P/R3K2R b KQ - 0 15",
        "3rr1k1/2p2p2/pp1q4/3N1p2/1P1PP2P/8/P7/2R1Q2K w - - 0 33",
        "8/8/4K1p1/7p/5k1N/8/8/8 b - - 4 79",
        "r1bq3r/ppp5/2kp1N2/2bn2N1/4P3/8/P2n1PPP/1RR3K1 w - - 0 22",
        "r4k1r/p1pq1ppp/3b4/4N3/4b3/2P1P3/P2P1PPP/R1B1K2R w KQ - 0 14",
        "8/1p6/p1p1PQ2/4Rb1p/P4pqk/1P6/3r2PP/6K1 b - - 11 52",
        "r3qb1r/pppbk1p1/2np2np/4p2Q/2BPP3/2P5/PP3PPP/RNB2RK1 w - - 4 11",
        "r1bq3r/ppp1nQ2/2kp1N2/6N1/3bP3/8/P2n1PPP/1RR3K1 b - - 1 20",
        "r1bq3r/ppp1nQ2/2kp1N2/2b3N1/4P3/8/P2n1PPP/1RR3K1 w - - 2 21",
        "kq1rr3/pp4pp/1Rp4n/Q1P2b2/R3B3/5N1P/P4PP1/6K1 w - - 1 28",
        "r1r3k1/1p2bppp/3p2q1/3Pp1PP/8/pPP1B1N1/P2n1P2/R3K1R1 w Q - 0 26",
        "rnbq2kr/ppp5/7p/3pB3/1b1Pn1pP/3B4/PPP3P1/R3QRK1 w - - 4 14",
        "2q1k2r/p1r1np1p/bp2p1p1/3pP1Q1/3n1P2/2P4N/P1BB1RPP/2R3K1 w k - 0 21",
        "5r1k/1p2q2p/p3Prp1/2pp3n/5n1Q/2P1B3/PPB4P/4R1RK w - - 0 33",
        "r5k1/pbp3pp/1p2pr2/5pn1/3P3q/1PPN1P2/1BP2QPP/R3R1K1 b - - 4 18",
        "r1bq4/ppppk3/2n1prB1/4b3/7Q/8/P4PPP/1NB2RK1 w - - 3 19",
        "r3r3/pR3pk1/q3b2p/3p4/3P2pP/2N3P1/P1PQ3K/5R2 w - - 4 22",
        "r2q1b1r/pppbk1p1/2np2np/4p2Q/2BPP3/2P5/PP3PPP/RNB2RK1 b - - 3 10",
        "3r3k/1pq2P2/p1p5/2P1pQ1p/7p/2P5/P4PP1/6K1 w - - 0 42",
        "5n2/2nk2Q1/4p2P/3pP3/1NpP1N1q/p1P5/3K4/8 b - - 14 60",
        "3r1r1k/b1p2pp1/p6p/3n3q/2NP4/P2Q1PB1/5P1P/1R2R1K1 b - - 2 26",
        "rnbq2kr/ppp5/7p/3pB3/1b1P2pP/2nB4/PPP3P1/R3QRK1 b - - 3 13",
        "8/8/2N2b2/8/P2p2Pp/1P1k3K/8/8 w - - 0 57",
        "2q1k2r/p1r1np1p/bp2p1p1/3pP1Q1/3P1P2/7N/P1BB1RPP/2R3K1 b k - 0 21",
        "r4k2/pb4p1/np2Qrq1/7p/2P1N3/3B4/PP3PPP/3R1RK1 w - - 3 22",
        "r3qn1k/ppbn1Rpp/2p1r3/3pP2Q/3P4/3B3P/PPPB2P1/4R2K w - - 7 21",
        "4rr1k/ppp3p1/7p/2qPnpNQ/8/1P4P1/1P3P1P/R3R1K1 w - - 0 25",
        "3r3k/7p/2p1np2/4p1p1/1Pq1P3/2Q2P2/P4RNP/2R4K b - - 0 42",
        "r1r3k1/1p2bppp/3p2q1/3Pp1PP/4n3/pPP1B1N1/P2Q1P2/R3K1R1 b Q - 0 25",
        "r1bq1rk1/pp1n2pN/4p1n1/3pPp1Q/3P1P1P/3B4/PP1B2P1/4K2R b K - 1 15",
        "2b1q2k/8/p2r3r/1p1p1pRQ/2pP1N2/4P3/PP6/1B4RK b - - 2 36",
        "3r3k/1pq2P2/p1p5/2P1pQpp/7N/2P5/P4PP1/6K1 b - - 0 41",
        "r3q1kr/ppp4p/3p2nQ/7P/3PP1b1/5R2/PPP3P1/5RK1 w - - 1 19",
        "r1bq3r/ppp1nQ2/2kp1N2/6N1/3bP3/8/P2n1PPP/1R3RK1 w - - 0 20",
        "r1bq3r/ppp1nQ2/1bkp1N2/6N1/3PP3/8/P2n1PPP/1R3RK1 b - - 6 19",
        "r1bq1r2/ppppk3/2n1p1B1/4b3/7Q/8/P4PPP/1NB2RK1 b - - 2 18",
        "2r2rk1/pp1q1ppp/3bpn2/3p1P2/1P1Q4/P1N1P3/1BP2P1P/R3K2R w KQ - 0 15",
        "5rk1/p5pp/8/8/2P1p3/1Pb3P1/7P/3R1N1K b - - 2 37",
        "r2q1r1k/ppp4p/5ppQ/1B2p3/3n2bb/2N1B3/PPP2P1P/R2R2K1 w - - 0 18",
        "r1bq3r/ppp1nkpp/1bnp4/3N2B1/3PP3/5N2/P4PPP/R2Q1RK1 w - - 2 13",
        "1k5r/pp2r3/8/P1bB4/2Pq1PQ1/1R4RP/1P6/7K w - - 12 40",
        "2r4k/pb2q2P/1p6/3Pp3/4p3/1P2R3/PBrQ2PP/5RK1 w - - 0 28",
        "3rn2k/ppq1r3/4Bp2/2pPp1p1/2P2bPp/PP3Q1P/1B2RP2/4R2K w - - 6 39",
        "6r1/p4p2/3k4/6N1/1ppP3P/1PPn1K2/P7/5R2 b - - 1 34",
        "2r4k/pb2q2P/1p2p3/3P4/4p3/1P2R3/PBrQ2PP/5RK1 b - - 0 27",
        "2r4k/pb2q2P/1p2p3/8/3Pp3/1P2R3/PBrQ2PP/5RK1 w - - 1 27",
        "1r1r2k1/5p1p/p1qp2p1/2pNp1b1/2P1P1P1/3Q3P/PP3P2/RR4K1 w - - 4 24",
        "5kn1/6p1/3q1b2/4pP2/pP1pP1N1/3P1N1Q/1rr5/5RRK b - - 1 49",
        "5nQ1/2nk4/4p2P/3pP3/1NpP1N1q/p1P5/3K4/8 w - - 13 60",
        "2r2rk1/1p4pp/pnb5/2Q2p2/5q2/N1P5/PPB2PPP/3R1RK1 b - - 0 19",
        "R7/p4pk1/1p2r3/4P2p/5Pr1/P3PK2/6p1/2R5 w - - 0 50",
        "5kn1/6p1/3q1b2/4pP2/1P1pP1N1/p2P1N1Q/1rr5/5RRK w - - 0 50",
        "2b1q2k/6R1/p2r3r/1p1p1p1Q/2pP1N2/4P3/PP6/1B4RK w - - 1 36",
        "2kr4/p2b1p2/1q6/1p2rP2/2p1B1P1/4R2p/PP1Q3P/3R2K1 w - - 0 32",
        "r2q1n1k/ppbn1Rpp/2p1r3/3pP2Q/3P4/3B3P/PPPB2P1/4R2K b - - 6 20",
        "2b1q1k1/6R1/p2r3r/1p1p1p1Q/2pP1N2/4P3/PP6/1B4RK b - - 0 35",
        "2b1q1k1/6b1/p2r3r/1p1p1p1Q/2pP1N2/4P1R1/PP6/1B4RK w - - 5 35",
        "r3k2r/ppp1qpp1/7p/2p1n3/2B3b1/3P4/PPPQ1PPP/4RRK1 b kq - 1 13",
        "4rr1k/ppp3pp/8/2qPnpNQ/8/1P4P1/1P3P1P/R3R1K1 b - - 4 24",
        "r7/5pk1/4p1p1/b2q4/p1pPpPN1/PrBbP3/1P1Q2K1/R6R w - - 1 40",
    ]

    # network_name = (
    #    # "T785469-600469c425eaf7397138f5f9edc18f26dfaf9791f365f71ebc52a419ed24e9f2"
    #    "T807301-c85375d37b369db8db6b0665d12647e7a7a3c9453f5ba46235966bc2ed433638"
    # )
    network_name = None
    plot_graph = False
    score_type = "q"

    engine_config_name = "remote_25_depth_stockfish.ini"
    search_limit = {"depth": 40}
    # engine_config_name = "remote_debug_500_nodes.ini"
    # engine_config_name = "remote_400_nodes.ini"
    # search_limit = {"nodes": 400}
    ################
    #  CONFIG END  #
    ################

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
            score_type=score_type,
        )
    )

    print("Results:")
    index = 0
    for fen, score in zip(fens, scores):
        print(f"board {fen :<74} score: {score}")
        if plot_graph:
            tree = trees[index]
            graph = convert_tree_to_networkx(tree, only_basic_info=True)
            print(f"Number of nodes: {len(graph.nodes)}")
            print(f"Number of edges: {len(graph.edges)}")
            stylized_network, config = nw.visualize(graph)
        index += 1
