from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import chess
import networkx as nx
import numpy as np

# from chess.engine import AnalysisResult
if TYPE_CHECKING:
    from rl_testing.engine_generators.relaxed_uci_protocol import ExtendedAnalysisResult


class TreeParser:
    # Every line containing node/tree info should begin with this
    # string.
    PARSE_TOKEN = "TREE INFO"
    START_TREE_TOKEN = "START TREE"
    END_TREE_TOKEN = "END TREE"
    START_NODE_TOKEN = "START NODE"
    END_NODE_TOKEN = "END NODE"
    NODE_TOKEN = "node"
    POSITION_TOKEN = "POSITION:"

    attribute_map = {
        "N": "num_visits",
        "move": "move",
        "IN_FLIGHT": "in_flight_visits",
        "P": "policy_value",
        "WL": "w_minus_l",
        "D": "draw_value",
        "M": "num_moves_left",
        "Q": "q_value",
        "U": "u_value",
        "S": "s_value",
        "V": "v_value",
    }

    def __init__(self, analysis_result: "ExtendedAnalysisResult") -> None:
        self.tree: Optional[TreeInfo] = None
        self.node: Optional[NodeInfo] = None
        self.node_cache: dict[str, NodeInfo] = {}
        self.analysis_result = analysis_result
        self.node_index = 0
        self.node_already_visited = False
        self.node_counter = 0
        self.node_duplicate_counter = 0
        self.num_edges_parsed = 0

    def parse_line(self, line: str) -> None:
        # Remove the PARSE_TOKEN and everything before it from the line.
        line = line.split(self.PARSE_TOKEN, 1)[1]

        if self.START_TREE_TOKEN in line:
            self.start_tree()
        elif self.END_TREE_TOKEN in line:
            self.end_tree()
        elif self.START_NODE_TOKEN in line:
            self.start_node()
        elif self.END_NODE_TOKEN in line:
            self.end_node()
        elif self.POSITION_TOKEN in line:
            self.parse_position(line)
        elif self.NODE_TOKEN in line:
            self.parse_node_line(line)
        else:
            self.parse_edge_line(line)

    def start_tree(self) -> None:
        self.tree = TreeInfo()

    def end_tree(self) -> None:
        assert self.tree is not None
        assert self.tree.root_node is not None

        # Compute the depth of each node
        num_per_depth = []
        self.tree.root_node.assign_depth(0, num_per_depth)

        # Store the node cache
        self.tree.node_cache = self.node_cache

        # Add the tree to the analysis result
        self.analysis_result.mcts_tree = self.tree
        for index in range(len(self.analysis_result.multipv)):
            self.analysis_result.multipv[index]["mcts_tree"] = self.tree

        # Print some stats
        # print(f"Node counter: {self.node_counter}")
        # print(f"Node duplicate counter: {self.node_duplicate_counter}")
        # print(f"Difference: {self.node_counter - self.node_duplicate_counter}")

        # Reset the parser
        self.tree = None
        self.node_index = 0
        self.node_cache = {}
        self.node_counter = 0
        self.node_duplicate_counter = 0

    def start_node(self) -> None:
        self.node = NodeInfo(self.node_index)
        self.num_edges_parsed = 0
        self.node_index += 1

        if self.tree.root_node is None:
            self.tree.root_node = self.node

        self.node_counter += 1

    def end_node(self) -> None:
        # If this node did not have any edges, then it can also act as a leaf node
        if (
            self.node_cache[self.node.fen].is_also_terminal
            and self.node_cache[self.node.fen].contains_only_terminal
            and self.num_edges_parsed > 0
        ):
            self.node_cache[self.node.fen].contains_only_terminal = False
        if self.num_edges_parsed == 0 and (not self.node_cache[self.node.fen].is_also_terminal):
            self.node_cache[self.node.fen].is_also_terminal = True
            self.node_cache[self.node.fen].contains_only_terminal = True

        self.node = None
        self.node_already_visited = False

    def parse_position(self, line: str) -> None:
        # Remove the POSITION_TOKEN and everything before it from the line.
        line = line.split(self.POSITION_TOKEN, 1)[1]

        # Parse the line if the line contains a fen string.
        if "/" in line:
            self.node.set_fen(line.strip())

        # Parse the line if the line contains a list of moves.
        elif "+" in line:
            root_board = chess.Board(self.tree.root_node.fen)
            move_list = line.split("+")
            move_list = [move.strip() for move in move_list]
            move_list = move_list[1:]
            for move in move_list:
                root_board.push_uci(move)

            self.node.set_fen(root_board.fen(en_passant="fen"))

            # Connect the node to its parent
            root_board.pop()
            parent_node = self.node_cache[root_board.fen(en_passant="fen")]

            if self.node.fen in self.node_cache:
                parent_node.connect_child_node(self.node_cache[self.node.fen])
            else:
                parent_node.connect_child_node(self.node)

            del root_board

        if self.node.fen in self.node_cache:
            self.node_index -= 1
            self.node_duplicate_counter += 1
            self.node_already_visited = True
        else:
            self.node_cache[self.node.fen] = self.node

    def parse_data_line(self, line: str) -> None:
        # Remove the node or edge token
        line = line.strip()
        line = line[line.index("(") + 1 : -1]
        if line[-1] == "T":
            line = line[:-4]
        tokens = line.split(") (")

        result_dict = {}
        for token in tokens:
            key, value = token.split(":")
            key, value = key.strip(), value.strip()

            if value.endswith("%"):
                value = float(value[:-1]) / 100
            elif value.startswith("+"):
                value = int(value[1:])
            elif "-.-" in value:
                value = None
            elif key == "move":
                if value != "node":
                    value = chess.Move.from_uci(value)
                else:
                    value = None
            else:
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError(f"Can't parse value {value}")

            if value is not None:
                result_dict[self.attribute_map[key]] = value

        return result_dict

    def parse_node_line(self, line: str) -> None:
        if not self.node_already_visited:
            attribute_dict = self.parse_data_line(line)

            for attribute in attribute_dict:
                setattr(self.node, attribute, attribute_dict[attribute])

            self.node.check_required_attributes()
        elif self.node_already_visited and self.node_cache[self.node.fen].contains_only_terminal:
            attribute_dict = self.parse_data_line(line)

            for attribute in attribute_dict:
                setattr(self.node_cache[self.node.fen], attribute, attribute_dict[attribute])

            self.node_cache[self.node.fen].check_required_attributes()

    def parse_edge_line(self, line: str) -> None:
        if not self.node_already_visited:
            attribute_dict = self.parse_data_line(line)

            # Create a new edge
            edge = EdgeInfo(attribute_dict["move"], self.node)
            del attribute_dict["move"]

            for attribute in attribute_dict:
                setattr(edge, attribute, attribute_dict[attribute])

            edge.check_required_attributes()
        elif self.node_already_visited and self.node_cache[self.node.fen].contains_only_terminal:
            attribute_dict = self.parse_data_line(line)

            # Create a new edge
            edge = EdgeInfo(attribute_dict["move"], self.node_cache[self.node.fen])
            del attribute_dict["move"]

            for attribute in attribute_dict:
                setattr(edge, attribute, attribute_dict[attribute])

            edge.check_required_attributes()

        self.num_edges_parsed += 1


class Info:
    required_attributes = []

    def __init__(self) -> None:
        # Initialize the node data
        self.num_visits: Optional[int] = None
        self.in_flight_visits: Optional[int] = None
        self.policy_value: Optional[float] = None
        self.w_minus_l: Optional[float] = None
        self.draw_value: Optional[float] = None
        self.num_moves_left: Optional[int] = None
        self.q_value: Optional[float] = None
        self.u_value: Optional[float] = None
        self.s_value: Optional[float] = None
        self.v_value: Optional[float] = None

    def check_required_attributes(self):
        for attribute_name in self.required_attributes:
            if getattr(self, attribute_name) is None:
                raise AttributeError(f"Attribute {attribute_name} must be set")


class TreeInfo:
    def __init__(self) -> None:
        self.root_node: Optional[NodeInfo] = None
        self.node_cache: dict[str, NodeInfo] = {}


class NodeInfo(Info):

    required_attributes = [
        "num_visits",
        "in_flight_visits",
        "policy_value",
        "w_minus_l",
        "draw_value",
        "num_moves_left",
        "q_value",
        "v_value",
        "is_also_terminal",
        "contains_only_terminal",
    ]

    def __init__(self, index: int) -> None:
        super().__init__()
        # Initialize parent and child edges
        self.parent_edges: List[EdgeInfo] = []
        self.child_edges: List[EdgeInfo] = []

        # Initialize the board position
        self.fen: Optional[str] = None
        self.index = index

        # Initialize depth information
        self.depth = -1
        self.depth_index = -1

        # This value indicates whether this node can also be a terminal node.
        self.is_also_terminal = False
        self.contains_only_terminal = False

    @property
    def child_nodes(self) -> List["NodeInfo"]:
        return [edge.end_node for edge in self.child_edges if edge.end_node is not None]

    @property
    def parent_nodes(self) -> List["NodeInfo"]:
        return [edge.start_node for edge in self.parent_edges if edge.start_node is not None]

    @property
    def orphan_edges(self) -> List["NodeInfo"]:
        return [edge for edge in self.child_edges if edge.end_node is None]

    def set_fen(self, fen: str) -> None:
        # Check if the fen string is valid.
        temp_board = chess.Board(fen)
        if temp_board.is_valid():
            self.fen = fen
        else:
            raise ValueError(f"Fen string {fen} is not valid.")

    def connect_child_node(self, child_node: "NodeInfo") -> None:
        board = chess.Board(self.fen)

        # Find the edge that connects the child node
        for edge in self.child_edges:
            board.push(edge.move)
            if board.fen(en_passant="fen") == child_node.fen:
                if edge.end_node is None:
                    edge.set_end_node(child_node)
                return
            board.pop()

        raise ValueError(f"Can't find edge to connect {child_node.fen} to {self.fen}.")

    def assign_depth(self, depth: int, num_per_depth: List[int]):
        # Assign your own depth
        self.depth = depth

        # Compute how many other nodes already have this depth
        if len(num_per_depth) <= depth:
            num_per_depth.append(0)
        self.depth_index = num_per_depth[depth]
        num_per_depth[depth] += 1

        # Assign the depths of the child nodes
        for edge in self.child_edges:
            if edge.end_node is not None:
                edge.end_node.assign_depth(max(depth + 1, edge.end_node.depth), num_per_depth)


class EdgeInfo(Info):
    required_attributes = [
        "start_node",
        "num_visits",
        "in_flight_visits",
        "policy_value",
        "q_value",
        "u_value",
        "s_value",
    ]

    def __init__(
        self,
        move: Union[str, chess.Move],
        start_node: Optional[NodeInfo] = None,
        end_node: Optional[NodeInfo] = None,
    ) -> None:
        super().__init__()
        # Initialize the start and end nodes
        self.start_node: Optional[NodeInfo] = None
        self.end_node: Optional[NodeInfo] = None

        if start_node is not None:
            self.set_start_node(start_node)
        if end_node is not None:
            self.set_end_node(end_node)

        # Initialize the move
        if isinstance(move, str):
            self.move: chess.Move = chess.Move.from_uci(move)
        elif isinstance(move, chess.Move):
            self.move = move
        else:
            raise TypeError(f"Move must be a string or chess.Move, not {type(move)}")

    def set_start_node(self, node: NodeInfo) -> None:
        self.start_node = node
        node.child_edges.append(self)

    def set_end_node(self, node: NodeInfo) -> None:
        self.end_node = node
        node.parent_edges.append(self)


def convert_tree_to_networkx(tree: TreeInfo, only_basic_info: bool = False) -> nx.DiGraph:
    red = np.array([255, 0, 0])
    green = np.array([0, 255, 0])
    white = np.array([255, 255, 255])

    graph = nx.Graph()
    # Add all nodes to the graph
    for fen in tree.node_cache:
        node = tree.node_cache[fen]

        if node.v_value is None:
            print("This should not happen!")

        # Compute the color of the new node
        node_value_current_player = node.v_value if node.depth % 2 == 0 else -node.v_value
        if node_value_current_player <= 0:
            color = red + (white - red) * (1 + node_value_current_player)
        else:
            color = green + (white - green) * (1 - node_value_current_player)
        color = color.round().astype(int)
        color_str = f"#{color[0]:0{2}x}{color[1]:0{2}x}{color[2]:0{2}x}"

        graph.add_node(
            node.index if only_basic_info else node,
            color=color_str,
            x=node.depth_index * 30,
            y=node.depth * 5,
        )
    for fen in tree.node_cache:
        node = tree.node_cache[fen]
        for edge in node.child_edges:
            if edge.end_node is not None:
                if only_basic_info:
                    assert edge.start_node.index in graph
                    assert edge.end_node.index in graph
                    graph.add_edge(edge.start_node.index, edge.end_node.index, size=edge.q_value)
                else:
                    assert edge.start_node in graph
                    assert edge.end_node in graph
                    graph.add_edge(edge.start_node, edge.end_node, size=edge.q_value)

    return graph
