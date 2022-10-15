import hashlib
import math
import random
from typing import List


class State():
    """_summary_

        Args:
            NUM_TRUNS: left depth of network.
            GOAL: close to 0 is better
            MOVES: choices provided
            MAX_VALUE: used to normalize the reward score
            num_moves: number of choices
            self.moves: current choices list
            self.value: current status
    """
    NUM_TURNS = 10
    GOAL = 0
    MOVES = [2, -2, 3, -3]
    MAX_VALUE = (5.0 * (NUM_TURNS - 1) * NUM_TURNS) / 2
    num_moves = len(MOVES)

    def __init__(self,
                 value: int = 0,
                 moves: List = None,
                 turn: int = NUM_TURNS):
        if moves is None:
            moves = []
        self.value = value
        self.turn = turn
        self.moves = moves

    def next_state(self):
        """generate next states by current node"""
        nextmove = random.choice([x * self.turn for x in self.MOVES])
        return State(self.value + nextmove, self.moves + [nextmove],
                     self.turn - 1)

    def terminal(self):
        """terminate when move the leaf node."""
        return self.turn == 0

    def reward(self):
        """reward functions."""
        return 1.0 - (abs(self.value - self.GOAL) / self.MAX_VALUE)

    def __hash__(self):
        return int(
            hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return 'Value: %d; Moves: %s' % (self.value, self.moves)


class Node():
    """Node of MC Tree.

    Args:
        state (_type_): states of current node.
        parent (_type_, optional): parenet node of current node.
            Defaults to None.
    """

    def __init__(self, state, parent=None):
        self.visits = 1  # number of visited, used for normalize UTC score
        self.reward = 0.0  # reward of current node
        self.state = state  # states of current node
        self.children = []  # list of children node
        self.parent = parent  # record parent node

    def add_child(self, child_state):
        """add child node for current node."""
        child = Node(child_state, self)
        self.children.append(child)

    def update(self, reward):
        """No function called.??"""
        self.reward += reward
        self.visits += 1

    def fully_expanded(self, num_moves_lambda):
        """judge whether currnet node is fully expanded."""
        # num_moves means number of choices.
        if num_moves_lambda is None:
            num_moves = self.state.num_moves
        else:
            num_moves = num_moves_lambda(self)
        return len(self.children) == num_moves

    def __repr__(self):
        return f'Node; children: {len(self.children)}; visits: {self.visits}; reward: {self.reward}'


class MCTS_Trainer:

    def __init__(self):
        super().__init__()

    def uct_search(self, simu_times, root: Node, num_choices: int = None):
        ...

    def free_policy(self, node: Node, num_chocies: int = None) -> Node:
        ...

    def expand_node(self, node: Node) -> Node:
        ...

    def best_child_node(self, node: Node, scalar: float = None) -> Node:
        if scalar is None:
            # hparams for adjusting explore weight
            scalar = 1 / (2 * math.sqrt(2.0))

    def default_policy(self, state) -> float:
        ...

    def back_propagation(self, node, reward) -> None:
        ...
