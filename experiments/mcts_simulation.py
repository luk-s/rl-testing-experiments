import math as m
from typing import List, Tuple


def initialize_parameters(
    p1: float,
) -> Tuple[float, float, float, float, float, float, float]:
    assert 0 <= p1 <= 1
    p2 = 1 - p1
    w1, w2 = 0, 0
    q1, q2 = 0, 0
    n1, n2 = 0, 0
    return p2, w1, w2, q1, q2, n1, n2


def choose_action(
    q1: float, q2: float, c_puct: float, p1: float, p2: float, n1: float, n2: float
) -> Tuple[str, float, float, float, float]:
    exploration_value1 = c_puct * p1 * m.sqrt(n1 + n2) / (1 + n1)
    exploration_value2 = c_puct * p2 * m.sqrt(n1 + n2) / (1 + n2)
    a1 = q1 + exploration_value1
    a2 = q2 + exploration_value2

    if a1 >= a2:
        action_string = "a1"
    else:
        action_string = "a2"

    return action_string, a1, a2, exploration_value1, exploration_value2


def update_parameters(
    action: str,
    n1: float,
    n2: float,
    w1: float,
    w2: float,
    q1: float,
    q2: float,
    value: float,
) -> Tuple[float, float, float, float, float, float]:
    if action == "a1":
        n1 += 1
        w1 += value
        q1 = w1 / n1
    elif action == "a2":
        n2 += 1
        w2 += value
        q2 = w2 / n2

    return n1, n2, w1, w2, q1, q2


def interactive_simulation(c_puct: float, p1: float) -> None:
    p2, w1, w2, q1, q2, n1, n2 = initialize_parameters(p1)

    finished = False
    while not finished:
        print("\n==================================================================\n")
        print(f"P(s,a1) = {p1}, Q(s,a1) = {q1}, W(s,a1) = {w1}, N(s,a1) = {n1}")
        print(f"P(s,a2) = {p2}, Q(s,a1) = {q2}, W(s,a2) = {w2}, N(s,a2) = {n2}\n")
        (
            action,
            action_value1,
            action_value2,
            exploration1,
            exploration2,
        ) = choose_action(q1, q2, c_puct, p1, p2, n1, n2)
        if action == "a1":
            print(
                f"Selected action a1. Exploration values: a1: {exploration1: .4f}, a2: {exploration2: .4f}"
            )
        elif action == "a2":
            print(
                f"Selected action a2. Exploration values: a1: {exploration1: .4f}, a2: {exploration2: .4f}"
            )
        print(f"a1 value = {action_value1}, a2 value = {action_value2}\n")

        value_str = ""
        value = None
        while True:
            value_str = input("Choose a value from [-1, 1]: ")
            try:
                if value_str == "quit":
                    finished = True
                else:
                    value = float(value_str)
                    assert -1 <= value <= 1
                break
            except ValueError:
                print("You need to input a float value!")
            except AssertionError:
                print("You need to input a value from [-1, 1]!")

        if value_str == "quit":
            break

        n1, n2, w1, w2, q1, q2 = update_parameters(
            action, n1, n2, w1, w2, q1, q2, value
        )


def offline_simulation(c_puct: float, p1: float, values: List[float]) -> List[str]:
    p2, w1, w2, q1, q2, n1, n2 = initialize_parameters(p1)
    actions_chosen = []

    for value in values:
        print("\n==================================================================")
        print(f"P(s,a1) = {p1}, Q(s,a1) = {q1}, W(s,a1) = {w1}, N(s,a1) = {n1}")
        print(f"P(s,a2) = {p2}, Q(s,a1) = {q2}, W(s,a2) = {w2}, N(s,a2) = {n2}\n")
        action, action_value1, action_value2 = choose_action(
            q1, q2, c_puct, p1, p2, n1, n2
        )

        if action == "a1":
            print("Selected action a1")
        elif action == "a2":
            print("Selected action a2")
        print(f"a1 value = {action_value1}, a2 value = {action_value2}\n")

        n1, n2, w1, w2, q1, q2 = update_parameters(
            action, n1, n2, w1, w2, q1, q2, value
        )
        actions_chosen.append(action)

    return actions_chosen


if __name__ == "__main__":
    # Config
    c_puct = 0.1
    p1 = 0.3

    values = [0.5] * 1000

    # interactive_simulation(c_puct=c_puct, p1=p1)
    actions = offline_simulation(c_puct=c_puct, p1=p1, values=values)
    print(actions)
