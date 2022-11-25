from typing import Union

import chess


def remove_pawns(board: chess.Board) -> Union[chess.Board, str]:
    piece_map = board.piece_map()
    positions = list(piece_map.keys())
    for position in positions:
        if piece_map[position].symbol() in ["P", "p"]:
            del piece_map[position]

    new_board = chess.Board()
    new_board.set_piece_map(piece_map)
    new_board.turn = board.turn

    if not new_board.is_valid():
        return "failed"

    return new_board


def rotate_90_clockwise(board: chess.Bitboard) -> chess.Bitboard:
    return chess.flip_vertical(chess.flip_diagonal(board))


def rotate_180_clockwise(board: chess.Bitboard) -> chess.Bitboard:
    return rotate_90_clockwise(rotate_90_clockwise(board))


def rotate_270_clockwise(board: chess.Bitboard) -> chess.Bitboard:
    return rotate_180_clockwise(rotate_90_clockwise(board))


if __name__ == "__main__":
    board = chess.Board("6rk/1bp1q2p/p4p1Q/1p1ppPP1/3bP3/2NP3R/PPP3r1/1K1R4 b - - 0 24")
    board = remove_pawns(board)

    for i in range(4):
        print(board.fen())
        board = board.transform(rotate_90_clockwise)

    print(board.transform(chess.flip_vertical).fen())
