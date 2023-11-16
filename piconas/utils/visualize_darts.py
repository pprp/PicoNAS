import argparse
import os

import igraph
import numpy as np
import pygraphviz as pgv
import torch
from tqdm import tqdm

# define a adjacent matrix of straight networks
s0_adj = torch.LongTensor(
    [
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
    ]
)


def read_feature(emb_path):
    dataset = torch.load(emb_path)
    feature = []
    test_acc = []
    for i in tqdm(range(len(dataset)), desc='load feature'):
        feature.append(dataset[i]['feature'].detach().numpy())
        test_acc.append(dataset[i]['test_accuracy'])
    feature = np.stack(feature, axis=0)
    test_acc = np.stack(test_acc, axis=0)
    return feature, test_acc


def adj2graph(ops, adj):
    if ops.dim() == 2:
        ops = ops.unsqueeze(0)
        adj = adj.unsqueeze(0)
    batch_size, _, _ = ops.shape
    node_ops = torch.argmax(ops, dim=2).numpy()
    ops_value = torch.max(ops, dim=2).values.numpy()
    node_num = []
    node_ops_nonzero = []
    # delete zero operation for nasbench 101
    for i, (op, val) in enumerate(zip(node_ops, ops_value)):
        node_ops_nonzero.append(op[val == 1].tolist())
        node_num.append(np.sum(val).item())
    adj = torch.triu(adj, diagonal=1)
    G = [
        igraph.Graph(
            node_num[i],
            torch.nonzero(adj[i]).tolist(),
            vertex_attrs={'operation': node_ops_nonzero[i]},
            directed=True,
        )
        for i in range(batch_size)
    ]
    return G


"""Network visualization"""


def plot_DAG(g, res_dir, name, data_type, backbone=False):
    # backbone: puts all nodes in a straight line
    file_name = os.path.join(res_dir, name + '.png')
    draw_network(g, file_name, data_type, backbone)
    return file_name


def draw_network(g, path, data_type, backbone=False):
    graph = pgv.AGraph(
        directed=True, strict=True, fontname='Helvetica', arrowtype='open'
    )
    if g is None:
        add_node(graph, 0, 0)
        graph.layout(prog='dot')
        graph.draw(path)
        return
    for idx in range(g.vcount()):
        add_node(graph, idx, g.vs[idx]['operation'], data_type)
    straight_edges = []
    for idx in range(g.vcount()):
        for node in g.get_adjlist(igraph.IN)[idx]:
            if node == idx - 1 and backbone:
                graph.add_edge(node, idx, weight=1)
                straight_edges.append((node, idx))
            else:
                graph.add_edge(node, idx, weight=0)
    all_straight_edges = [(i, i + 1) for i in range(g.vcount() - 1)]
    diff_straight = list(set(all_straight_edges) - set(straight_edges))
    if diff_straight:
        for e in diff_straight:
            graph.add_edge(
                e[0], e[1], color='white'
            )  # white edges doesn't appear in graph, which controls shape
    graph.layout(prog='dot')
    graph.draw(path)


def add_node(
    graph, node_id, label, data_type='nasbench101', shape='box', style='filled'
):
    if data_type == 'nasbench101':
        if label == 0:
            label = 'in'
            color = 'skyblue'
        elif label == 1:
            label = '1x1'
            color = 'pink'
        elif label == 2:
            label = '3x3'
            color = 'yellow'
        elif label == 3:
            label = 'MP'
            color = 'orange'
        elif label == 4:
            label = 'out'
            color = 'beige'
    elif data_type == 'nasbench201':
        if label == 0:
            label = 'in'
            color = 'skyblue'
        elif label == 1:
            label = '1x1'
            color = 'pink'
        elif label == 2:
            label = '3x3'
            color = 'yellow'
        elif label == 3:
            label = 'pool'
            color = 'orange'
        elif label == 4:
            label = 'skip'
            color = 'greenyellow'
        elif label == 5:
            label = 'none'
            color = 'seagreen3'
        elif label == 6:
            label = 'out'
            color = 'beige'
    elif data_type == 'darts':
        pass
    else:
        print('do not support!')
        exit()
    label = f'{label}'
    graph.add_node(
        node_id,
        label=label,
        color='black',
        fillcolor=color,
        shape=shape,
        style=style,
        fontsize=24,
    )


def get_straight(dataset, num=1):
    # find a straight network
    idx = []
    for i in tqdm(range(len(dataset)), desc='find {} straight nets'.format(num)):
        tmp = torch.LongTensor(dataset[str(i)]['module_adjacency'])
        if torch.all(s0_adj == tmp):
            idx.append(i)
    idx = np.stack(idx)
    if num == 1:
        return [np.random.choice(idx, num).tolist()]
    else:
        return np.random.choice(idx, num).tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Networks (Graph)')
    parser.add_argument(
        '--data_type',
        type=str,
        default='nasbench101',
        help='benchmark type (default: nasbench101)',
        choices=['nasbench101', 'nasbench201', 'darts'],
        metavar='TYPE',
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='data *.json file (default: None)',
        metavar='PATH',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./output',
        help='output path (default: None)',
        metavar='PATH',
    )
    parser.add_argument(
        '--random_path',
        type=int,
        default=50,
        help='num of paths to visualization (default: 50)',
        metavar='N',
    )
    parser.add_argument(
        '--path_step',
        type=int,
        default=10,
        help='num of points of each visualization (default: 10)',
        metavar='N',
    )
    parser.add_argument(
        '--straight_path',
        type=int,
        default=10,
        help='num of paths starting at a straight networks (default: 10)',
        metavar='N',
    )
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    output_path = os.path.join(
        args.output_path, args.data_type, '{}steps'.format(args.path_step)
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, 'compare')):
        os.makedirs(os.path.join(output_path, 'compare'))

    adj = torch.tensor(
        [
            [0, 1, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
    )

    ops = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=torch.int32,
    )

    G = adj2graph(ops, adj)
    file_name = plot_DAG(G[0], os.path.curdir, 'example', 'nasbench201')
