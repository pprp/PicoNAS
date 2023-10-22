import random
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import networkx as nx


def sample_combination_bitwise():
    combinations = [
        0b001,  # G
        0b010,  # W
        0b100,  # X
        0b011,  # G, W
        0b101,  # G, X
        0b110,  # W, X
        0b111  # G, W, X
    ]
    idx = random.randint(0, 6)
    bit_combination = combinations[idx]
    result = []
    if bit_combination & 0b001: result.append('G')
    if bit_combination & 0b010: result.append('W')
    if bit_combination & 0b100: result.append('X')
    return result


import networkx as nx


class Graph(nx.DiGraph):

    def __init__(self):
        super().__init__()

    def has_cycle(self):
        """Checks if the graph contains a cycle."""
        return nx.algorithms.dag.has_cycle(self)

    def is_reachable(self, start, end):
        """Checks if one node is reachable from another."""
        return nx.algorithms.dag.is_reachable(self, start, end)

    def mutate(self):
        """Randomly applies one of three mutations to the graph: add an edge, remove an edge, or reverse an edge."""
        mutation_type = random.choice(['add', 'remove', 'reverse'])

        if mutation_type == 'add':
            node1 = random.choice(list(self.nodes))
            node2 = random.choice(list(self.nodes))
            if node1 != node2 and node2 not in self.adj[node1]:
                self.add_edge(node1, node2)
                if nx.algorithms.dag.has_cycle(self):
                    self.remove_edge(node1, node2)

        elif mutation_type == 'remove':
            node1 = random.choice(list(self.nodes))
            if node1 in self.adj:
                node2 = random.choice(list(self.adj[node1]))
                self.remove_edge(node1, node2)

        elif mutation_type == 'reverse':
            node1 = random.choice(list(self.nodes))
            if node1 in self.adj:
                node2 = random.choice(list(self.adj[node1]))
                self.remove_edge(node1, node2)
                self.add_edge(node2, node1)
                if nx.algorithms.dag.has_cycle(self):
                    self.remove_edge(node2, node1)
                    self.add_edge(node1, node2)

    @staticmethod
    def crossover(parent1, parent2):
        """Combines two parent graphs to produce a child graph, ensuring that the child graph remains a DAG."""
        split_node = random.choice(list(parent1.nodes))

        child = Graph()

        # Add nodes from parents to child
        for node in parent1.nodes:
            child.add_node(node)
        for node in parent2.nodes:
            child.add_node(node)

        # Copy edges from parent1 up to split_node
        for node in parent1.nodes:
            if node == split_node:
                break
            for neighbor in parent1.adj[node]:
                child.add_edge(node, neighbor)

        # Copy edges from parent2 from split_node onwards
        split_reached = False
        for node in parent2.nodes:
            if node == split_node:
                split_reached = True
            if split_reached:
                for neighbor in parent2.adj[node]:
                    child.add_edge(node, neighbor)

        # Check if the child graph has a cycle due to crossover
        # In a real-world scenario, we might want to handle this more gracefully
        # Here, we simply attempt to mutate the child graph to fix it
        while nx.algorithms.dag.has_cycle(child):
            child.mutate()

        return child

    def visualize(self):
        """Visualizes the graph using the NetworkX library."""
        nx.draw(
            self,
            pos=nx.spring_layout(self),
            with_labels=True,
            node_size=2000,
            node_color='skyblue',
            font_size=15,
            width=2,
            alpha=0.6)
        plt.title('Graph Visualization')
        # plt.show()
        plt.savefig('./visualize_graph.png')

    def adjacency_matrix(self):
        """Returns an adjacency matrix representation of the graph."""
        return nx.adjacency_matrix(self).toarray()

    def __repr__(self) -> str:
        """Matrix representation of the graph."""
        matrix = self.adjacency_matrix()
        return '\n'.join(
            [' '.join([str(cell) for cell in row]) for row in matrix])


    def find_path(self, start_nodes, end_node):
        """Finds a path from one of the start nodes to the end node in the given graph.

        Args:
            graph: A networkx graph.
            start_nodes: A list of start nodes.
            end_node: The end node.

        Returns:
            A list of nodes in the path, or None if no path is found.
        """

        visited = set()
        stack = [(start_node, []) for start_node in start_nodes]

        while stack:
            current_node, path = stack.pop()

            if current_node not in visited:
                visited.add(current_node)
                path.append(current_node)

            if current_node == end_node:
                return path

            for neighbor in self.neighbors(current_node):
                stack.append((neighbor, path.copy()))
        return None

def create_dag_graph(input_combination):
    print('Selected input combination: ', input_combination)
    graph = Graph()
    operations = ['Op1', 'Op2', 'Op3', 'Op4', 'Op5', 'Op6']
    output_node = 'Output'

    # Add nodes for inputs, operations, and output
    for inp in input_combination:
        graph.add_node(inp)
    for operation in operations:
        graph.add_node(operation)
    graph.add_node(output_node)

    # Ensure the graph is a DAG using topological sort
    while True:
        graph = Graph()
        # Add nodes for inputs, operations, and output
        for inp in input_combination:
            graph.add_node(inp)
        for operation in operations:
            graph.add_node(operation)
        graph.add_node(output_node)

        for _ in range(3):
            # Randomly connect inputs to operations
            for inp in input_combination:
                chosen_op_for_input = random.choice(operations)
                graph.add_edge(inp, chosen_op_for_input)

        # Randomly connect one operation to the output
        chosen_op_for_output = random.choice(operations)
        graph.add_edge(chosen_op_for_output, output_node)

        # Ensure the graph is a DAG using topological sort
        is_dag = nx.is_directed_acyclic_graph(graph)
        if is_dag:
            break

    # Print the graph structure
    print(graph)

    return graph


# Sample usage
input_combination = sample_combination_bitwise()
dag_graph = create_dag_graph(input_combination)
dag_graph.visualize()

# Get the adjacency matrix
matrix = dag_graph.adjacency_matrix()

# Print the adjacency matrix
for row in matrix:
    print(row)

# find path from input_combination to output
path = dag_graph.find_path(input_combination, 'Output')
print('Path: ', path)

