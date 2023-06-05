import pandas as pd
import chess


def remove_bad_forced_moves(data_path: str) -> pd.DataFrame:
    # Load the data
    df = pd.read_csv(data_path, header=None, names=["fen"])

    invalid_row_indices = []

    # Iterate over all rows of the dataframe
    for row_index, row in df.iterrows():
        if row_index % 1000 == 0:
            print(f"Processing row {row_index}/{len(df)}")

        # Do sanity check on the FEN
        board = chess.Board(row["fen"])

        # Check that there is only one legal move for the current player
        if len(list(board.legal_moves)) != 1:
            print(f"Found FEN with more than one legal move: {row_index}: {row['fen']}")
            invalid_row_indices.append(row_index)
            continue

        # Check that after playing the only legal move, the game is not over
        board.push(list(board.legal_moves)[0])
        if board.is_game_over():
            print(
                f"Found FEN with game over after playing the only legal move: {row_index}:"
                f" {row['fen']}"
            )
            invalid_row_indices.append(row_index)

    # Remove all invalid rows
    df = df.drop(invalid_row_indices)

    return df


def remove_already_processed_fens(
    dataframe: pd.DataFrame, fen_cache_path: str, fen_column: str, header: bool = True
) -> pd.DataFrame:
    # Load the FEN cache
    if not header:
        fen_cache = pd.read_csv(fen_cache_path, header=None, names=[fen_column])
    else:
        fen_cache = pd.read_csv(fen_cache_path)

    # Remove all FENs that are already present in the "parent_fen" column of the FEN cache
    dataframe = dataframe[~dataframe["fen"].isin(fen_cache[fen_column])]

    return dataframe


if __name__ == "__main__":
    # data_path = "/data/chess-data/all_forced_move_positions.txt"
    data_path = "/data/chess-data/all_forced_move_positions_end.txt"

    fen_cache_path = "temp/results_ENGINE_local_400_nodes_DATA_all_forced_move_positions_fen_2023_05_07_21:34:15.txt"  # noqa
    fen_cache_path2 = "temp/all_forced_move_positions_cleaned_part2.txt"

    df = remove_bad_forced_moves(data_path)

    # Print the number of rows in the dataframe
    print(f"Number of rows in the dataframe: {len(df)}")

    # Remove all FENs that are already present in the FEN cache
    df = remove_already_processed_fens(df, fen_cache_path, "parent_fen")

    # Print the number of rows in the dataframe
    print(f"Number of rows in the dataframe: {len(df)}")

    df = remove_already_processed_fens(df, fen_cache_path2, "fen", header=False)

    # Print the number of rows in the dataframe
    print(f"Number of rows in the dataframe: {len(df)}")

    # Store the dataframe
    df.to_csv("/data/chess-data/all_forced_move_positions_end2.txt", index=False, header=False)
