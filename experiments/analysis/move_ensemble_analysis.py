from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from load_results import load_data


class EdgeMetric:
    __slots__ = [
        "move",
        "num_visits",
        "in_flight_visits",
        "policy_value",
        "v_value",
        "q_value",
        "u_value",
        "s_value",
    ]

    def __init__(
        self,
        move: str,
        num_visits: int,
        in_flight_visits: int,
        policy_value: float,
        v_value: float,
        q_value: float,
        u_value: float,
        s_value: float,
    ):
        self.move = move
        self.num_visits = num_visits
        self.in_flight_visits = in_flight_visits
        self.policy_value = policy_value
        self.v_value = v_value
        self.q_value = q_value
        self.u_value = u_value
        self.s_value = s_value

    def search_key(self) -> str:
        return self.move


class Metrics:
    __slots__ = [
        "fen",
        "score",
        "edges",
    ]

    def __init__(self, fen: str, score: float, edges: List[EdgeMetric]):
        self.fen = fen
        self.score = score
        self.edges = edges


def build_score_dict(df: pd.DataFrame) -> Metrics:
    score_dict = {}
    for index, row in df.iterrows():
        # Extract the columns from the row
        fen = row["fen"]
        score = row["score"]
        if score == "invalid":
            continue
        edge_info = row["edge_info"][:-1]
        if "None" in edge_info:
            continue

        # Parse the edge_info string into a list of EdgeMetric objects
        edge_metrics = []
        edge_list = edge_info.split(",")
        assert len(edge_list) % 8 == 0
        for i in range(0, len(edge_list), 8):
            edge_metrics.append(
                EdgeMetric(
                    move=edge_list[i],
                    num_visits=int(float(edge_list[i + 1])),
                    in_flight_visits=int(float(edge_list[i + 2])),
                    policy_value=float(edge_list[i + 3]),
                    v_value=float(edge_list[i + 4]),
                    q_value=float(edge_list[i + 5]),
                    u_value=float(edge_list[i + 6]),
                    s_value=float(edge_list[i + 7]),
                )
            )

        # Sort the edge_metrics by their search key
        edge_metrics.sort(key=lambda x: x.search_key())

        # Add a new Metrics object to the dictionary
        score_dict[fen] = Metrics(fen=fen, score=score, edges=edge_metrics)

    return score_dict


def ensemble_prediction_most_visits(score_dicts: List[Dict[str, Metrics]]) -> Dict[str, str]:
    # Compute the total number of visits for each move
    result_dict = {}
    # Iterate over the different score dictionaries
    for score_dict in score_dicts:

        # Iterate over all fens in the score dictionary
        for fen, metrics in score_dict.items():
            if fen not in result_dict:
                result_dict[fen] = {}

            # Iterate over all moves possible from the fen position
            for edge_metric in metrics.edges:
                result_dict[fen][edge_metric.move] = (
                    result_dict[fen].get(edge_metric.move, 0) + edge_metric.num_visits
                )

    # Find the move with the most visits for each fen
    prediction_dict = {}
    for fen, move_dict in result_dict.items():
        prediction_dict[fen] = max(move_dict, key=move_dict.get)

    return prediction_dict


if __name__ == "__main__":
    ensemble_file_names = [
        Path("results_ENGINE_local_100_nodes_debug_DATA_database_NET_T608927.txt"),
        Path("results_ENGINE_local_100_nodes_debug_DATA_database_NET_T785469.txt"),
        Path("results_ENGINE_local_100_nodes_debug_DATA_database_NET_T611246.txt"),
        Path("results_ENGINE_local_100_nodes_debug_DATA_database_NET_T807301.txt"),
        Path("results_ENGINE_local_100_nodes_debug_DATA_database_NET_T771717.txt"),
    ]
    single_file_name = "results_ENGINE_local_500_nodes_debug_DATA_database_NET_T785469.txt"
    result_folder = Path(__file__).parent.parent / Path("results/tree_stat_testing")

    # Create the list of result dictionaries for the ensemble files
    ensemble_score_dicts = []
    for ensemble_file_name in ensemble_file_names:
        print(f"Loading data from {ensemble_file_name}...")
        df, config = load_data(result_folder / ensemble_file_name, separator=";")
        df = df.dropna()
        ensemble_score_dicts.append(build_score_dict(df))

    # Create an ensemble prediction dictionary
    print("Creating ensemble prediction dictionary...")
    ensemble_prediction_dict = ensemble_prediction_most_visits(ensemble_score_dicts)

    # Create a list of result dictionaries for the single file
    print(f"Loading data from {single_file_name}...")
    df, config = load_data(result_folder / single_file_name, separator=";")
    df = df.dropna()
    single_score_dict = build_score_dict(df)

    # Create a single prediction dictionary
    print("Creating single prediction dictionary...")
    single_prediction_dict = ensemble_prediction_most_visits([single_score_dict])

    # Compare the predictions of the ensemble and the single model
    print("Comparing predictions...")
    different_fens = []
    for fen, ensemble_move in ensemble_prediction_dict.items():
        if fen in single_prediction_dict and ensemble_move != single_prediction_dict[fen]:
            different_fens.append((fen, ensemble_move, single_prediction_dict[fen]))

    # Print the number of different fens
    print(f"Number of different fens: {len(different_fens)}")

    # Store the different fens in a csv file
    print("Storing different fens in csv file...")
    df = pd.DataFrame(
        different_fens,
        columns=["fen", "ensemble_move", "single_move"],
    )
    df.to_csv(result_folder / "different_fens.csv", index=False)

    print("Finished!")
