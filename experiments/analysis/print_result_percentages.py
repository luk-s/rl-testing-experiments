import argparse
import math as m
from pathlib import Path
from typing import List, Tuple

import pandas as pd


# DATA_PATH = Path("experiments/results/final_data/")
def get_num_samples_and_score_columns(file_name: str) -> Tuple[int, List[str]]:
    if "400k" in file_name:
        num_samples = 400_000
    elif "200k" in file_name:
        num_samples = 200_000
    elif "100k" in file_name:
        num_samples = 100_000
    elif "50k" in file_name:
        num_samples = 50_000

    if "evolutionary_algorithm" in file_name:
        score_columns = ["fitness1", "fitness2"]
    elif "50k" in file_name and "board_transformation" in file_name:
        score_columns = ["original", "rot180"]
    elif "board_transformation" in file_name:
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
    elif "mirror" in file_name:
        score_columns = ["original", "mirror"]
    elif "recommended_move" in file_name or "forced_move" in file_name:
        score_columns = ["score1", "score2"]
    elif "differential_testing" in file_name:
        score_columns = ["score1", "score2"]
    else:
        raise ValueError(f"Unknown file name: {file_name}")

    return num_samples, score_columns


def get_percentages(raw_row: str, cutoff: float = 0.01) -> str:
    numbers = raw_row.split(" & ")
    original_divisor = numbers[0]

    numbers[0] = numbers[0].replace("k", "000")
    divisor = numbers[0]

    numbers = [int(number) for number in numbers]

    divisor = numbers[0]

    def round_to_1(x):
        if x == 0:
            return 0
        if x > 1:
            return round(x, 1)
        return round(x, -int(m.floor(m.log10(abs(x)))))

    def to_string(x):
        if x == 0:
            return "0\%"
        if x < cutoff:
            return f"<{cutoff}\%"
        return f"{x}\%"

    numbers = [round_to_1(100 * (number / divisor)) for number in numbers]

    numbers = [original_divisor] + [to_string(number) for number in numbers[1:]]
    return " & ".join(numbers)


def get_numbers(
    file_name: str, score_columns: List[str], num_samples: int, bins: List[float]
) -> str:
    # Load csv file
    df = pd.read_csv(file_name)

    # Compute the maximum difference between any two scores in the score columns
    max_diff = df[score_columns].max(axis=1) - df[score_columns].min(axis=1)

    # For each bin, compute the number of differences which are larger than the value of the bin
    num_diffs = []
    for bin in bins:
        num_diffs.append((max_diff > bin).sum())

    return f"{num_samples // 1000}k & " + " & ".join([str(num_diff) for num_diff in num_diffs])


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, help="Path to result file", required=True)  # noqa
    parser.add_argument("--mode", type=str, help="Mode", choices=["numbers", "percentages"], default="percentages")  # noqa
    parser.add_argument("--cutoff", type=float, help="Cutoff for percentages", default=0.01)  # noqa
    # fmt: on
    args = parser.parse_args()

    file_name = args.result_path

    bins = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

    num_samples, score_columns = get_num_samples_and_score_columns(file_name)

    number_row = get_numbers(file_name, score_columns, num_samples, bins)

    if args.mode == "numbers":
        print(number_row)
    elif args.mode == "percentages":
        percentage_row = get_percentages(number_row, cutoff=args.cutoff)
        print(percentage_row)
