import hashlib
import logging
import math
import queue
import random
from typing import List

from graphviz import Digraph

logging.basicConfig(level=logging.INFO)

GLOBAL_ID_FOR_STATE = 1


class State():
    """_summary_

        Args:
            NUM_TRUNS: left depth of network.
            GOAL: close to 0 is better
            status: choices provided
            MAX_VALUE: used to normalize the reward score
            num_choices: number of choices
            self.status: current choices list
            self.value: current status

        left_depth is ranging from [0, 6)
    """

    def __init__(self,
                 value: int = 0,
                 status: List = None,
                 left_depth: int = 6,
                 current_choice=None):
        if status is None:
            # current status
            status = []

        global GLOBAL_ID_FOR_STATE
        self.unique_id = GLOBAL_ID_FOR_STATE
        GLOBAL_ID_FOR_STATE += 1
        self.current_choice = str(
            current_choice) if current_choice is not None else 'root'
        self.MAX_DEPTHS = 6
        self.CHOICE_KEYS = ['att', 'ms', 'local', 'batch', 'channel', 'kl']
        self.CHOICES = {
            'att': [True, False],
            'ms': [True, False],
            'local': [True, False],
            'batch': [True, False],
            'channel': [True, False],
            'kl': [True, False],
        }
        self.NUM_CHOICES = {
            'att': 2,
            'ms': 2,
            'local': 2,
            'batch': 2,
            'channel': 2,
            'kl': 2
        }

        self.value = value
        self.left_depth = left_depth
        self.status = status

        # self.distiller = GLOBAL_DISTILLER

        # update value
        current_reward = self.reward()
        self.value += current_reward

    def next_state(self):
        """generate next states by current node"""
        key_of_next_depth = self.CHOICE_KEYS[self.left_depth - 1]
        next_choice = random.choice(self.CHOICES[key_of_next_depth])
        return State(
            self.value,
            self.status + [next_choice],
            self.left_depth - 1,
            current_choice=next_choice)

    def terminal(self):
        """terminate when move the leaf node."""
        return self.left_depth == 1

    def reward(self):
        """reward functions."""
        # loss_fn = self.build_loss_fn()
        # return self.distiller.estimate_rewards(loss_fn)
        return random.random()

    def __hash__(self):
        return int(
            hashlib.md5(str(self.status).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        res = f'Node(value={self.value}, depth={self.left_depth}):'
        for k, v in zip(self.CHOICE_KEYS[:len(self.status)], self.status):
            res += f'\nkey: {k} \t Chosen: {v}'
        res += '\n---------------'
        return res

    @property
    def node_id(self):
        """Unique id"""
        left_depth = self.left_depth
        c_key = self.CHOICE_KEYS[self.MAX_DEPTHS - left_depth]
        return f'{c_key}_{self.current_choice}_d_{self.left_depth}_id_{self.unique_id}'


class Node():
    """Node of MC Tree.

    Args:
        state (_type_): states of current node.
        parent (_type_, optional): parenet node of current node.
            Defaults to None.
    """

    def __init__(self, state: State, parent=None):
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

    def fully_expanded(self, num_choices=None):
        """judge whether currnet node is fully expanded."""
        # num_choices means number of choices.
        left_depth = self.state.left_depth
        c_key = self.state.CHOICE_KEYS[self.state.MAX_DEPTHS - left_depth]
        num_choices = self.state.NUM_CHOICES[
            c_key] if num_choices is None else num_choices
        return len(self.children) == num_choices

    def __repr__(self):
        return f'Node=(children: {len(self.children)}, visits: {self.visits}, reward: {self.reward})'

    @property
    def node_info(self):
        left_depth = self.state.left_depth
        c_key = self.state.CHOICE_KEYS[self.state.MAX_DEPTHS - left_depth]
        return f'{c_key}:{self.state.current_choice} \n R: {self.reward/self.visits:.1f}'


class MCTS_Trainer:

    def __init__(self, scalar=None):
        super().__init__()
        if scalar is None:
            # MCTS scalar.  Larger scalar will increase exploitation,
            # smaller will increase exploration.
            self.scalar = 1 / (2 * math.sqrt(2.0))
        self.logger = logging.getLogger('MCTS')

    def uct_search(self, root_node: Node, simu_times: int):
        """search function of UCT

        Args:
            simu_times (int): the number of simulations to perform, which
                should be a large number.
            root (Node): _description_
        """
        for iter in range(simu_times):
            front_node = self.tree_policy(root_node)
            reward = self.default_policy(front_node.state)
            self.back_propagation(front_node, reward)
            self.logger.info(
                f'Iter: {iter} Status of current front: {front_node.state}')

        return self.best_child(root_node, 0)

    def tree_policy(self, node: Node) -> Node:
        """
        a hack to force 'exploitation' in a game where there are many options,
        and you may never/not want to fully expand_node first
        """
        while not node.state.terminal():
            if len(node.children) == 0:
                # For root node and leaf node.
                return self.expand_node(node)
            elif random.uniform(0, 1) < .5:
                # exploitation with random policy for middle node
                # 50% get the best_child
                node = self.best_child(node, self.scalar)
            else:
                # For middle node
                if not node.fully_expanded():
                    return self.expand_node(node)
                else:
                    node = self.best_child(node, self.scalar)
        return node

    def expand_node(self, node: Node) -> Node:
        """expand one child node for current node."""
        tried_children = [c.state for c in node.children]
        # get children generated by `next_state`
        new_state = node.state.next_state()
        cnt = 0
        while new_state in tried_children and not new_state.terminal():
            new_state = node.state.next_state()
            cnt += 1
            if cnt > 100:
                import pdb
                pdb.set_trace()
        # append the new node to current node.
        node.add_child(new_state)
        return node.children[-1]

    def best_child(self, node: Node, scalar: float = None) -> Node:
        """get best child by UTC score."""
        bestscore = 0.0
        best_children = []
        for c in node.children:
            exploit = c.reward / c.visits
            explore = math.sqrt(2.0 * math.log(node.visits) / float(c.visits))
            # UTC score calculation
            score = exploit + scalar * explore
            if score == bestscore:
                best_children.append(c)
            if score > bestscore:
                best_children = [c]
                bestscore = score

        if len(best_children) == 0:
            self.logger.warning('OOPS: no best child found, probably fatal')
        return random.choice(best_children)

    def default_policy(self, state) -> float:
        """Judge whether to terminate and return rewards."""
        while not state.terminal():
            state = state.next_state()
        return state.reward()

    def back_propagation(self, node, reward) -> None:
        """back propagation and update the reward of nodes."""
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent


def show_search_tree(root):
    # https://zhuanlan.zhihu.com/p/333348915
    dot = Digraph(
        comment='Game Search Tree',
        engine='dot',
        node_attr={
            'color': 'lightblue2',
            'style': 'filled'
        })
    dot.format = 'png'
    if isinstance(root, Node):
        dot.node(root.state.node_id, root.node_info)

        que = queue.Queue()
        que.put(root.children[0])

        while not que.empty():
            child = que.get()
            if isinstance(child, Node):
                dot.node(child.state.node_id, child.node_info)
                dot.edge(child.parent.state.node_id, child.state.node_id)
                for c in child.children:
                    que.put(c)
    with open('a.dot', 'w', encoding='utf-8') as writer:
        writer.write(dot.source)
    dot.render('search_path', view=False)


def main():
    mcts_trainer = MCTS_Trainer()
    root_node = Node(State())
    current_node = mcts_trainer.uct_search(root_node, simu_times=10000)
    for i, c in enumerate(current_node.children):
        print(i, c)
    print(f'best child: {current_node.state}')

    show_search_tree(root_node)


if __name__ == '__main__':
    main()
