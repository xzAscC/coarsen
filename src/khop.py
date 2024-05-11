# 要记录的属性：节点数，边缘数，是否连通，是否有节点/边属性，平均度，多跳子图大小，训练集大小，测试集大小，是否有向
from logger import LightLogging
from tqdm import tqdm 
import torch_geometric as pyg
import matplotlib.pyplot as plt

def khop(graph_list, log,k=[2, 3]):
    log.info('Start recording...')  
    for graph_name in graph_list:
        # 节点数，边缘数，是否有节点/边属性，平均度，多跳子图大小，训练集大小，测试集大小，是否有向
        log.info('--------------------------------------------------------------------------')
        log.info('Start processing graph: %s' % graph_name)
        graph = pyg.datasets.Planetoid(root='../dataset', name=graph_name, split="public")[0]
        log.info(f'Number of nodes: {graph.num_nodes}')
        log.info(f'Number of edges: {graph.num_edges}')
        log.info(f'Is undirected: {graph.is_undirected()}')
        log.info(f'num_edge_features: {graph.num_edge_features}')
        log.info(f'num_node_features: {graph.num_node_features}')
        log.info(f'avg degree: {graph.num_edges / graph.num_nodes}')
        log.info(f'Number of train samples: {graph.test_mask.sum()}')
        log.info(f'Number of test samples: {graph.test_mask.sum()}')
        log.info(f'Number of val samples: {graph.val_mask.sum()}')
    
        for k_num in k:
            sub_graph_size = []
            sub_graph_edge_size = []
            for node_index in tqdm(range(graph.x.size(0))):
                subset, edge_index, mapping, edge_mask = pyg.utils.k_hop_subgraph([node_index], 2, graph.edge_index)
                sub_graph_size.append(subset.size(0))
                sub_graph_edge_size.append(edge_index.size(1))
            
            # visualize the size of subgraph
            plt.hist(sub_graph_size, bins=30)
            plt.xlabel(f'Size of {k_num}-hop subgraph')
            plt.ylabel('Frequency')
            plt.show()
            plt.savefig(f'../misc/{graph_name}_{k_num}_hop_subgraph_size.png')
            plt.hist(sub_graph_edge_size, bins=30)
            plt.xlabel(f'Size of {k_num}-hop subgraph edge')
            plt.ylabel('Frequency')
            plt.show()
            plt.savefig(f'../misc/{graph_name}_{k_num}_hop_subgraph_edge_size.png')
            log.info(f'avg {k_num}-hop subgraph size: {sum(sub_graph_size) / graph.num_nodes}')            
            log.info(f'max {k_num}-hop subgraph size: {max(sub_graph_size)}')
            log.info(f'min {k_num}-hop subgraph size: {min(sub_graph_size)}')
            log.info(f'avg {k_num}-hop subgraph edge size: {sum(sub_graph_edge_size) / graph.num_nodes}')
            log.info(f'max {k_num}-hop subgraph edge size: {max(sub_graph_edge_size)}')
            log.info(f'min {k_num}-hop subgraph edge size: {min(sub_graph_edge_size)}')
            
        log.info('--------------------------------------------------------------------------')


if __name__ == '__main__':
    log = LightLogging(log_path='../log', log_name='khop', log_level='info')
    graph_list = ['Cora', 'Citeseer', 'Pubmed']
    khop(graph_list, log)
