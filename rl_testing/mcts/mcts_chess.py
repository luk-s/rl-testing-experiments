import asyncio
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import numpy as np

from rl_testing.engine_generators.generators import EngineGenerator
from rl_testing.mcts.tree_parser import TreeInfo
from rl_testing.util.util import get_task_result_handler


class Parameters:
    def __init__(
        self,
        Cpuct=1.745,
        CpuctAtRoot=1.745,
        CpuctFactor=3.894,
        CpuctFactorAtRoot=3.894,
        CpuctBase=38739.0,
        CpuctBaseAtRoot=38739.0,
        FpuValue=0.330,
        FpuValueAtRoot=1.0,
        FpuStrategy="reduction",
        FpuStrategyAtRoot="same",
        DrawScoreSideToMove=0.0,
        DrawScoreOpponent=0.0,
        DrawScoreWhite=0.0,
        DrawScoreBalck=0.0,
        MovesLeftSupport=True,
        MovesLeftMaxEffect=0.0345,
        MovesLeftThreshold=0.0,
        MovesLeftSlope=0.0027,
        MovesLeftConstantFactor=0.0,
        MovesLeftScaledFactor=1.6521,
        MovesLeftQuadraticFactor=-0.6521,
    ) -> None:
        self.Cpuct = Cpuct
        self.CpuctAtRoot = CpuctAtRoot
        self.CpuctFactor = CpuctFactor
        self.CpuctFactorAtRoot = CpuctFactorAtRoot
        self.CpuctBase = CpuctBase
        self.CpuctBaseAtRoot = CpuctBaseAtRoot
        self.FpuValue = FpuValue
        self.FpuValueAtRoot = FpuValueAtRoot
        self.FpuStrategy = FpuStrategy
        self.FpuStrategyAtRoot = FpuStrategyAtRoot
        self.DrawScoreSideToMove = DrawScoreSideToMove / 100
        self.DrawScoreOpponent = DrawScoreOpponent / 100
        self.DrawScoreWhite = DrawScoreWhite / 100
        self.DrawScoreBlack = DrawScoreBalck / 100
        self.MovesLeftSupport = MovesLeftSupport
        self.MovesLeftMaxEffect = MovesLeftMaxEffect
        self.MovesLeftThreshold = MovesLeftThreshold
        self.MovesLeftSlope = MovesLeftSlope
        self.MovesLeftConstantFactor = MovesLeftConstantFactor
        self.MovesLeftScaledFactor = MovesLeftScaledFactor
        self.MovesLeftQuadraticFactor = MovesLeftQuadraticFactor

    def get_cpuct(self, at_root: bool) -> float:
        return self.CpuctAtRoot if at_root else self.Cpuct

    def get_cpuct_factor(self, at_root: bool) -> float:
        return self.CpuctFactorAtRoot if at_root else self.CpuctFactor

    def get_cpuct_base(self, at_root: bool) -> float:
        return self.CpuctBaseAtRoot if at_root else self.CpuctBase

    def get_fpu_absolute(self, at_root: bool) -> bool:
        return (
            (
                (self.FpuStrategyAtRoot == "same" and self.get_fpu_absolute(False))
                or self.FpuStrategyAtRoot == "absolute"
            )
            if at_root
            else self.FpuStrategy == "absolute"
        )

    def get_draw_score(self, is_odd_depth: bool, white_to_move: bool) -> float:
        # root is 0 i.e. even depth
        return (
            self.DrawScoreOpponent
            if is_odd_depth
            else self.DrawScoreSideToMove + self.DrawScoreWhite
            if white_to_move
            else self.DrawScoreBlack
        )


def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)


class ChessMovesLeftEvaluator:
    def __init__(self, params: Parameters) -> None:
        if params.MovesLeftSupport:
            self.enabled = True
            self._moves_left_max_effect = params.MovesLeftMaxEffect
            self._moves_left_threshold = params.MovesLeftThreshold
            self._moves_left_slope = params.MovesLeftSlope
            self._moves_left_constant_factor = params.MovesLeftConstantFactor
            self._moves_left_scaled_factor = params.MovesLeftScaledFactor
            self._moves_left_quadratic_factor = params.MovesLeftQuadraticFactor
        else:
            self.enabled = False
            self._moves_left_max_effect = 0.0
            self._moves_left_threshold = 0.0
            self._moves_left_slope = 0.0
            self._moves_left_constant_factor = 0.0
            self._moves_left_scaled_factor = 0.0
            self._moves_left_quadratic_factor = 0.0

        self.parent = None
        self.parent_within_threshold = False
        self.parent_moves_left_estimate = 0.0

    def set_parent_node(self, parent: "ChessSearchNode"):
        self.parent = parent
        self.parent_within_threshold = self.within_threshold()
        self.parent_moves_left_estimate = self.parent._num_moves_left_estimated

    def within_threshold(self) -> bool:
        return abs(self.parent.q_value(0.0)) > self._moves_left_threshold

    def get_default_M(self) -> float:
        return 0.0

    def get_M(self, moves_left_estimate: float, q_value: float) -> float:
        assert self.parent is not None
        if (not self.enabled) or (not self.parent_within_threshold):
            return 0.0

        m = clamp(
            self._moves_left_slope * (moves_left_estimate - self.parent_moves_left_estimate),
            -self._moves_left_max_effect,
            self._moves_left_max_effect,
        )
        m *= -1 if -q_value < 0 else 1
        m *= (
            self._moves_left_constant_factor
            + self._moves_left_scaled_factor * abs(q_value)
            + self._moves_left_quadratic_factor * q_value * q_value
        )
        return m


class EvaluationMetrics:
    __slots__ = [
        "value",
        "win_minus_loss",
        "draw_score",
        "num_moves_left_estimated",
    ]

    def __init__(
        self,
        value: float,
        win_minus_loss: float,
        draw_score: float,
        num_moves_left_estimated: float,
    ):
        self.value = value
        self.win_minus_loss = win_minus_loss
        self.draw_score = draw_score
        self.num_moves_left_estimated = num_moves_left_estimated


class ChessEvaluator:
    def evaluate(self, state) -> List[EvaluationMetrics]:
        """Returns evaluation on given state."""
        raise NotImplementedError

    def prior(self, state):
        """Returns a probability for each legal action in the given state."""
        raise NotImplementedError


class ChessEnsembleEvaluator(ChessEvaluator):
    def __init__(self, logger: logging.Logger) -> None:
        self.engine_generators = None
        self.network_names = None
        self.input_queues = None
        self.output_queues = None
        self.analysis_tasks = None
        self.is_async = True
        self.initialized = False
        self.analysis_failed = False
        self.logger = logger

        self.result_cache: Dict[str, List[TreeInfo]] = {}

    async def initialize_engine_tasks(
        self,
        engine_generators: List[EngineGenerator],
        network_names: Optional[List[str]] = None,
        sleep_after_get: float = 0.05,
    ) -> None:
        num_engines = len(engine_generators)
        if network_names is None:
            network_names = [None] * num_engines

        # Create the engine tasks
        self.input_queues = [asyncio.Queue() for _ in range(num_engines)]
        self.output_queues = [asyncio.Queue() for _ in range(num_engines)]

        # Create the analysis tasks
        self.analysis_tasks = [
            asyncio.create_task(
                self._analyze_position(
                    consumer_queue=input_queue,
                    producer_queue=output_queue,
                    engine_generator=engine_generator,
                    network_name=network_name,
                    sleep_after_get=sleep_after_get,
                    identifier_str=f"ENGINE_{index}",
                )
            )
            for index, engine_generator, network_name, input_queue, output_queue in zip(
                range(num_engines),
                engine_generators,
                network_names,
                self.input_queues,
                self.output_queues,
            )
        ]

        # Add callbacks to all tasks
        handle_task_exception = get_task_result_handler(
            logger=self.logger, message="Task raised an exception"
        )
        for task in self.analysis_tasks:
            task.add_done_callback(handle_task_exception)

        self.initialized = True

    async def shutdown_engine_tasks(self) -> None:
        if self.initialized:
            # Cancel all tasks
            for task in self.analysis_tasks:
                task.cancel()

            self.initialized = False

    async def _analyze_position(
        self,
        consumer_queue: asyncio.Queue,
        producer_queue: asyncio.Queue,
        engine_generator: EngineGenerator,
        network_name: Optional[str] = None,
        sleep_after_get: float = 0.0,
        identifier_str: str = "",
    ) -> None:
        """Returns evaluation on given state."""
        # Initialize the engine
        if network_name is not None:
            engine_generator.set_network(network_name=network_name)
        engine = await engine_generator.get_initialized_engine()

        while True:
            # Fetch the next base board, the next transformed board, and the corresponding
            # transformation index
            board = await consumer_queue.get()
            fen = board.fen(en_passant="fen")
            await asyncio.sleep(delay=sleep_after_get)

            logging.info(f"[{identifier_str}] Analyzing board {fen}: ")
            try:
                # Analyze the board
                info = await engine.analyse(board, chess.engine.Limit(nodes=1))
                assert "mcts_tree" in info
            except chess.engine.EngineTerminatedError:

                # Try to kill the failed engine
                logging.info(f"[{identifier_str}] Trying to kill engine")
                engine_generator.kill_engine(engine=engine)

                # Try to restart the engine
                logging.info("Trying to restart engine")
                if network_name is not None:
                    engine_generator.set_network(network_name=network_name)
                engine = await engine_generator.get_initialized_engine()

                # Set the failure flag
                await producer_queue.put("invalid")
            else:
                score_cp = info["score"].white().score(mate_score=12780)

                # Check if the computed score is valid
                if engine_generator is not None and not engine_generator.cp_score_valid(score_cp):
                    await producer_queue.put("invalid")
                else:
                    # Add the board to the list of analysis results
                    await producer_queue.put(info["mcts_tree"])
            finally:
                consumer_queue.task_done()

    async def analyze_position(self, board: chess.Board) -> List[Union[str, TreeInfo]]:
        fen = board.fen(en_passant="fen")

        if fen not in self.result_cache:
            # Send the board to the engines
            for input_queue in self.input_queues:
                await input_queue.put(board)

            # Wait for the results
            results = []
            for output_queue in self.output_queues:
                results.append(await output_queue.get())

            # Put the results in the cache
            self.result_cache[fen] = results
        else:
            results = self.result_cache[fen]

        return results

    async def evaluate(
        self, board: chess.Board
    ) -> Union[Tuple[EvaluationMetrics, EvaluationMetrics], str]:
        """Returns evaluation on given state."""
        analysis_results = await self.analyze_position(board=board)

        values, wins_minus_losses, draw_scores, num_moves_estimated = [], [], [], []

        for tree_info in analysis_results:
            if tree_info == "invalid":
                return "invalid"

            root_node = tree_info.root_node

            # Extract scores
            values.append(root_node.v_value)
            wins_minus_losses.append(root_node.w_minus_l)
            draw_scores.append(root_node.draw_value)
            num_moves_estimated.append(root_node.num_moves_left)

        current_metrics = EvaluationMetrics(
            value=np.mean(values),
            win_minus_loss=np.mean(wins_minus_losses),
            draw_score=np.mean(wins_minus_losses),
            num_moves_left_estimated=np.mean(num_moves_estimated),
        )

        opponent_metrics = EvaluationMetrics(
            value=-np.mean(values),
            win_minus_loss=-np.mean(wins_minus_losses),
            draw_score=-np.mean(wins_minus_losses),
            num_moves_left_estimated=np.mean(num_moves_estimated),
        )

        current_player_id = int(board.turn)
        opponent_player_id = 0 if current_player_id else 1
        result = [None, None]
        result[current_player_id] = current_metrics
        result[opponent_player_id] = opponent_metrics

        return result

    async def prior(self, board: chess.Board) -> Union[List[Tuple[int, float]], str]:
        """Returns a probability for each legal action in the given state."""
        analysis_results = await self.analyze_position(board=board)

        legal_move_dict = {}

        for tree_info in analysis_results:
            if tree_info == "invalid":
                return "invalid"

            # Get the priors of the legal actions
            edges = tree_info.root_node.child_edges
            for edge in edges:
                legal_move_dict[edge.move] = legal_move_dict.get(edge.move, 0) + edge.policy_value

        # Normalize the priors
        legal_move_priors = [
            (move, legal_move_dict[move] / len(analysis_results)) for move in legal_move_dict
        ]

        return legal_move_priors


class ChessSearchNode:
    __slots__ = [
        "parent",
        "action",
        "player",
        "prior",
        "depth",
        "draw_constant",
        "explore_count",
        "total_reward",
        "outcome",
        "children",
        "_is_root_node",
        "_win_minus_loss",
        "_draw_score",
        "_num_moves_left_estimated",
    ]

    parameters = Parameters()
    moves_left_evaluator = ChessMovesLeftEvaluator(parameters)

    def __init__(
        self,
        parent: "ChessSearchNode",
        action: str,
        player: int,
        prior: Optional[float],
        depth: int,
        is_root_node: bool,
    ):
        self.parent = parent
        self.action = action
        self.player = player
        self.prior = prior
        self.depth = depth
        self._num_moves_left_estimated = 0.0
        self._win_minus_loss = 0.0
        self._draw_score = 0.0
        self._is_root_node = is_root_node
        self.explore_count = 0
        self.total_reward = 0.0
        self.outcome = None
        self.children = []

        # Apparently, WHITE is 1 and BLACK is 0
        self.draw_constant = self.parameters.get_draw_score(depth % 2 == 1, self.player == 1)

    @staticmethod
    def set_parameters(parameters: Parameters):
        ChessSearchNode.parameters = parameters
        ChessSearchNode.moves_left_evaluator = ChessMovesLeftEvaluator(parameters)

    def q_value(self, draw_score=0) -> float:
        """Compute the Q value for this node.

        Returns:
            float: The Q value for this node.
        """
        return self._win_minus_loss + draw_score * self._draw_score

    def compute_cpuct(self, parent_visit_count: int) -> float:
        """Compute a constant for the U term in the MCTS formula.

        Args:
            parent_visit_count (int): The number of times the parent node has been visited.

        Returns:
            float: The constant for the U term in the MCTS formula.
        """
        return self.parameters.get_cpuct(self._is_root_node) + self.parameters.get_cpuct_factor(
            self._is_root_node
        ) * math.log(
            (parent_visit_count + self.parameters.get_cpuct_base(self._is_root_node))
            / self.parameters.get_cpuct_base(self._is_root_node)
        )

    def compute_fpu(self, draw_score) -> float:
        """Compute the first-play urgency value for this node.

        Returns:
            float: The first-play urgency value for this node.
        """
        value = self.parameters.get_fpu_absolute(self._is_root_node)
        if self.parameters.get_fpu_absolute(self._is_root_node):
            return value
        return -self.q_value(-draw_score) - value * math.sqrt(self.prior)

    def puct_value(self, parent_explore_count: int) -> float:
        """Compute the PUCT value for this node.

        Args:
            parent_explore_count (int): The number of times the parent node has been explored.

        Returns:
            float: The PUCT value for this node.
        """
        if self.outcome is not None:
            return self.outcome[self.player]

        # Compute a few constants
        const_puct = self.compute_cpuct(parent_explore_count)
        puct_multiplier = const_puct * math.sqrt(max(parent_explore_count - 1, 1))

        # Compute the value term of the PUCT formula
        self.moves_left_evaluator.set_parent_node(self.parent)
        if self.explore_count == 0:
            value = self.compute_fpu(self.draw_constant)
            moves_left = self.moves_left_evaluator.get_default_M()
        else:
            value = self.q_value(self.draw_constant)
            moves_left = self.moves_left_evaluator.get_M(self._num_moves_left_estimated, value)

        return self.prior * puct_multiplier / (1 + self.explore_count) + value + moves_left

    def sort_key(self):
        """Returns the best action from this node, either proven or most visited.

        This ordering leads to choosing:
        - Highest proven score > 0 over anything else, including a promising but
          unproven action.
        - A proven draw only if it has higher exploration than others that are
          uncertain, or the others are losses.
        - Uncertain action with most exploration over loss of any difficulty
        - Hardest loss if everything is a loss
        - Highest expected reward if explore counts are equal (unlikely).
        - Longest win, if multiple are proven (unlikely due to early stopping).
        """
        return (
            0 if self.outcome is None else self.outcome[self.player],
            self.explore_count,
            self.q_value(self.draw_constant),
        )

    def best_child(self):
        """Returns the best child in order of the sort key."""
        return max(self.children, key=ChessSearchNode.sort_key)

    def backpropagate_value(
        self, evaluation_metrics: Union[float, EvaluationMetrics], distance_to_leaf: int
    ):
        """Use the provided evaluation metrics to update the value of this node.

        Args:
            evaluation_metrics (EvaluationMetrics): A set of metrics to use to update the value of
                this node.
            distance_to_leaf (int): The distance of this node to the leaf node where the evaluation
                metrics were computed. This is necessary to update the number of moves left
        """
        # Check if this is a terminal node
        if isinstance(evaluation_metrics, float):
            win_minus_loss = evaluation_metrics
            draw_score = float(not evaluation_metrics)
            moves_left_estimated = 0.0
            value = evaluation_metrics
        else:
            win_minus_loss = evaluation_metrics.win_minus_loss
            draw_score = evaluation_metrics.draw_score
            moves_left_estimated = evaluation_metrics.num_moves_left_estimated + distance_to_leaf
            value = evaluation_metrics.value

        if self.explore_count == 0:
            self._win_minus_loss = win_minus_loss
            self._draw_score = draw_score
            self._num_moves_left_estimated = moves_left_estimated
        else:
            self._win_minus_loss += (win_minus_loss - self._win_minus_loss) / self.explore_count
            self._draw_score = (draw_score - self._draw_score) / self.explore_count
            self._num_moves_left_estimated = (
                moves_left_estimated - self._num_moves_left_estimated
            ) / self.explore_count
        self.total_reward += value

        self.explore_count += 1


class ChessMCTSBot:
    def __init__(
        self,
        max_simulations: int,
        evaluator: ChessEnsembleEvaluator,
        solve: bool = True,
        random_state=None,
        child_selection_fn=ChessSearchNode.puct_value,
        dirichlet_noise=None,
        verbose: bool = False,
        dont_return_chance_node: bool = False,
    ):
        """Initializes a MCTS Search algorithm in the form of a bot.

        In multiplayer games, or non-zero-sum games, the players will play the
        greedy strategy.

        Args:
          game: A pyspiel.Game to play.
          uct_c: The exploration constant for UCT.
          max_simulations: How many iterations of MCTS to perform. Each simulation
            will result in one call to the evaluator. Memory usage should grow
            linearly with simulations * branching factor. How many nodes in the
            search tree should be evaluated. This is correlated with memory size and
            tree depth.
          evaluator: A `Evaluator` object to use to evaluate a leaf node.
          solve: Whether to back up solved states.
          random_state: An optional numpy RandomState to make it deterministic.
          child_selection_fn: A function to select the child in the descent phase.
            The default is UCT.
          dirichlet_noise: A tuple of (epsilon, alpha) for adding dirichlet noise to
            the policy at the root. This is from the alpha-zero paper.
          verbose: Whether to print information about the search tree before
            returning the action. Useful for confirming the search is working
            sensibly.
          dont_return_chance_node: If true, do not stop expanding at chance nodes.
            Enabled for AlphaZero.

        Raises:
          ValueError: if the game type isn't supported.
        """
        self.max_simulations = max_simulations
        self.evaluator = evaluator
        self.verbose = verbose
        self.solve = solve
        self.max_utility = 1.0
        self._dirichlet_noise = dirichlet_noise
        self._random_state = random_state or np.random.RandomState()
        self._child_selection_fn = child_selection_fn
        self.dont_return_chance_node = dont_return_chance_node

    async def _apply_tree_policy(
        self, root: ChessSearchNode, board: chess.Board
    ) -> Union[Tuple[str, str], Tuple[List[ChessSearchNode], chess.Board]]:
        """Applies the UCT policy to play the game until reaching a leaf node.

        A leaf node is defined as a node that is terminal or has not been evaluated
        yet. If it reaches a node that has been evaluated before but hasn't been
        expanded, then expand it's children and continue.

        Args:
          root: The root node in the search tree.
          state: The state of the game at the root node.

        Returns:
          visit_path: A list of nodes descending from the root node to a leaf node.
          working_state: The state of the game at the leaf node.
        """
        visit_path = [root]
        working_board = board.copy()
        current_node = root
        while not working_board.is_game_over(claim_draw=True) and current_node.explore_count > 0:
            if not current_node.children:
                # For a new node, initialize its state, then choose a child as normal.
                legal_actions = await self.evaluator.prior(working_board)
                if legal_actions == "invalid":
                    return "invalid", "invalid"
                if current_node is root and self._dirichlet_noise:
                    epsilon, alpha = self._dirichlet_noise
                    noise = self._random_state.dirichlet([alpha] * len(legal_actions))
                    legal_actions = [
                        (a, (1 - epsilon) * p + epsilon * n)
                        for (a, p), n in zip(legal_actions, noise)
                    ]
                # Reduce bias from move generation order.
                self._random_state.shuffle(legal_actions)
                player = int(working_board.turn)
                current_node.children = [
                    ChessSearchNode(
                        parent=current_node,
                        action=action,
                        player=player,
                        prior=prior,
                        depth=current_node.depth + 1,
                        is_root_node=False,
                    )
                    for action, prior in legal_actions
                ]

            # Otherwise choose node with largest PUCT value
            chosen_child = max(
                current_node.children,
                key=lambda c: self._child_selection_fn(  # pylint: disable=g-long-lambda
                    c, current_node.explore_count
                ),
            )

            working_board.push(chosen_child.action)
            current_node = chosen_child
            visit_path.append(current_node)

        return visit_path, working_board

    async def mcts_search(self, board: chess.Board) -> Union[str, ChessSearchNode]:
        """A vanilla Monte-Carlo Tree Search algorithm.

        This algorithm searches the game tree from the given state.
        At the leaf, the evaluator is called if the game state is not terminal.
        A total of max_simulations states are explored.

        At every node, the algorithm chooses the action with the highest PUCT value,
        defined as: `Q/N + c * prior * sqrt(parent_N) / N`, where Q is the total
        reward after the action, and N is the number of times the action was
        explored in this position. The input parameter c controls the balance
        between exploration and exploitation; higher values of c encourage
        exploration of under-explored nodes. Unseen actions are always explored
        first.

        At the end of the search, the chosen action is the action that has been
        explored most often. This is the action that is returned.

        This implementation supports sequential n-player games, with or without
        chance nodes. All players maximize their own reward and ignore the other
        players' rewards. This corresponds to max^n for n-player games. It is the
        norm for zero-sum games, but doesn't have any special handling for
        non-zero-sum games. It doesn't have any special handling for imperfect
        information games.

        The implementation also supports backing up solved states, i.e. MCTS-Solver.
        The implementation is general in that it is based on a max^n backup (each
        player greedily chooses their maximum among proven children values, or there
        exists one child whose proven value is game.max_utility()), so it will work
        for multiplayer, general-sum, and arbitrary payoff games (not just win/loss/
        draw games). Also chance nodes are considered proven only if all children
        have the same value.

        Some references:
        - Sturtevant, An Analysis of UCT in Multi-Player Games,  2008,
          https://web.cs.du.edu/~sturtevant/papers/multi-player_UCT.pdf
        - Nijssen, Monte-Carlo Tree Search for Multi-Player Games, 2013,
          https://project.dke.maastrichtuniversity.nl/games/files/phd/Nijssen_thesis.pdf
        - Silver, AlphaGo Zero: Starting from scratch, 2017
          https://deepmind.com/blog/article/alphago-zero-starting-scratch
        - Winands, Bjornsson, and Saito, "Monte-Carlo Tree Search Solver", 2008.
          https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf

        Arguments:
          state: pyspiel.State object, state to search from

        Returns:
          The most visited move from the root node.
        """
        board.turn
        root = ChessSearchNode(
            parent=None,
            action=None,
            player=int(board.turn),
            prior=None,
            depth=0,
            is_root_node=True,
        )
        for _ in range(self.max_simulations):
            visit_path, working_board = await self._apply_tree_policy(root, board)
            if visit_path == "invalid":
                return "invalid"
            if working_board.is_game_over(claim_draw=True):
                outcome_info = working_board.outcome(claim_draw=True)
                winner = outcome_info.winner
                if winner is None:
                    returns = (0.0, 0.0)
                elif winner:
                    returns = (-1.0, 1.0)
                else:
                    returns = (1.0, -1.0)

                visit_path[-1].outcome = returns
                solved = self.solve
            else:
                returns = await self.evaluator.evaluate(working_board)
                if returns == "invalid":
                    return "invalid"
                solved = False

            distance_to_leaf = 0
            while visit_path:
                # For chance nodes, walk up the tree to find the decision-maker.
                decision_node_idx = -1
                # Chance node targets are for the respective decision-maker.
                return_values = returns[visit_path[decision_node_idx].player]
                node: ChessSearchNode = visit_path.pop()
                node.backpropagate_value(return_values, distance_to_leaf=distance_to_leaf)
                distance_to_leaf += 1

                if solved and node.children:
                    player = node.children[0].player
                    # If any have max utility (won?), or all children are solved,
                    # choose the one best for the player choosing.
                    best = None
                    all_solved = True
                    for child in node.children:
                        if child.outcome is None:
                            all_solved = False
                        elif best is None or child.outcome[player] > best.outcome[player]:
                            best = child
                    if best is not None and (
                        all_solved or best.outcome[player] == self.max_utility
                    ):
                        node.outcome = best.outcome
                    else:
                        solved = False
            if root.outcome is not None:
                break

        return root
