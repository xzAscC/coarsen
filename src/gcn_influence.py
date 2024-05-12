from logger import LightLogging
import torch
import torch.nn.functional as F   
import torch.nn as nn
import torch_geometric as pyg
from tqdm import tqdm
import matplotlib.pyplot as plt


class GcnInfluence(torch.nn.Module):
    def __init__(self, num_features, gcn_layers=4):
        super().__init__()
        self.gcn_layers = gcn_layers
        self.GCNs = pyg.nn.GCNConv(num_features, num_features)
        self.softmax = F.softmax
    
    def forward(self, graph, onehot):
        for i in range(self.gcn_layers):
            x = self.GCNs(onehot, graph.edge_index).relu()
        return self.softmax(x, dim=1)
    

if __name__ == '__main__':
    logger = LightLogging(log_path='../log', log_name='gcn_influence', log_level='info')
    logger.info('-' *50)
    logger.info('Start running gcn_influence.py')
    # design the dataset
    # max_degree of Cora is 168
    transform = pyg.transforms.Compose([
        pyg.transforms.OneHotDegree(max_degree=168, cat=False)
    ])
    dataset = pyg.datasets.Planetoid(root='../dataset', name='Cora', transform=transform)
    graph = dataset[0]
    # design a model
    # random or trained
    mode = 'random'
    # degree or distance
    influence_index = 'distance'
    gcn_layers = 4
    with torch.no_grad():
        if mode == 'random':
            if influence_index == 'degree':
                degree = pyg.utils.degree(dataset.edge_index[0], num_nodes=dataset[0].num_nodes)
                max_degree = int(degree.max().item())
                gcn_model = GcnInfluence(max_degree+1, gcn_layers=gcn_layers)

                subgraph = []
                mapping_list = []
                for node_index in tqdm(range(graph.num_nodes)):
                    k = 3
                    subset, edge_index, mapping, mask = pyg.utils.k_hop_subgraph(node_index, k, graph.edge_index, relabel_nodes=True)
                    cur_x = graph.x[subset]
                    cur_x[mapping] = torch.zeros_like(cur_x[mapping])
                    cur_subgraph = pyg.data.Data(x=cur_x, edge_index=edge_index, num_nodes=subset.size(0))
                    subgraph.append(cur_subgraph)
                    mapping_list.append(mapping)
                    # should have 1-hop node feature, edge_index and hop-label

                node_influence = []
                with tqdm(total=len(subgraph)) as pbar:
                    for idx, data in enumerate(subgraph):
                        out = gcn_model(data)
                        node_influence.append(out[mapping_list[idx]])
                        pbar.update(1)

                all_node_influence = node_influence[0]
                for i in tqdm(range(1, len(node_influence))):
                    all_node_influence += node_influence[i].numpy()
                
                all_node_influence = (all_node_influence / len(subgraph))[0]
                # visualize node_influence
                logger.info(f'all_node_influence shape: {all_node_influence.shape}')
                
                plt.plot(all_node_influence)
                plt.savefig('../misc/random_degree_influence.png')

            elif influence_index == 'distance':
                onehot = torch.eye(graph.num_nodes)
                logger.info(f'onehot shape: {onehot.shape}')
                gcn_model = GcnInfluence(onehot.size(1), gcn_layers=gcn_layers)

                out = gcn_model(graph, onehot)

                torch.save(out, '../misc/random_degree_influence_fullgraph_Cora.pt')

                # visualize node_influence
                logger.info(f'out shape: {out.shape}')
                subset, _,_,_ = pyg.utils.k_hop_subgraph(0, 2, graph.edge_index, relabel_nodes=True)
                plt.hist(out[0][subset])
                plt.savefig('../misc/random_degree_influence_2hop_Cora.png')
                subset, _,_,_ = pyg.utils.k_hop_subgraph(0, 1, graph.edge_index, relabel_nodes=True)
                plt.hist(out[0][subset])
                plt.savefig('../misc/random_degree_influence_1hop_Cora.png')
                subset, _,_,_ = pyg.utils.k_hop_subgraph(0, 3, graph.edge_index, relabel_nodes=True)
                plt.hist(out[0][subset])
                plt.savefig('../misc/random_degree_influence_3hop_Cora.png')
                #plt.hist(out)
                #plt.savefig('../misc/random_degree_influence_fullgraph_Cora.png')
        else:
            pass

    # running on the dataset and record the influence

    logger.info('End running gcn_influence.py')
    logger.info('-' *50)