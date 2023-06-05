from pathlib import Path
import pandas as pd

RESULT_PATH = Path(__file__).parent / Path("results/final_data")

if __name__ == "__main__":
    file_name = "transform_results_ENGINE_local_400_nodes_DATA_final_no_pawns_synthetic_fen_combined_differences_sorted.csv"  # noqa

    df = pd.read_csv(RESULT_PATH / Path(file_name))

    score_columns = [
        "original",
        "rot90",
        "rot180",
        "rot270",
        "flip_diag",
        "flip_anti_diag",
        "flip_hor",
        "flip_vert",
    ]

    counts = {name: 0 for name in score_columns}
    original_counts = {name: 0 for name in score_columns}

    # Iterate over the first 169 rows of the dataframe
    for row_index, row in df.iterrows():
        if row_index >= 1000:
            break

        # Extract all score columns
        scores = row[score_columns].tolist()

        # Get the argmin and argmax of the scores
        argmin = scores.index(min(scores))
        argmax = scores.index(max(scores))

        # Print the corresponding column names
        name_min = score_columns[argmin]
        name_max = score_columns[argmax]
        print(f"Row {row_index}:")
        print(f"Difference: {row['difference']}")
        print(f"Min: {name_min}")
        print(f"Max: {name_max}")
        print()

        counts[name_min] += 1
        counts[name_max] += 1

        if name_min == "original":
            original_counts[name_max] += 1
        elif name_max == "original":
            original_counts[name_min] += 1

    print(counts)
    print(original_counts)
