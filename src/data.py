from torch_geometric.data import InMemoryDataset, Dataset
from tqdm import tqdm
import torch
import os
import pandas as pd
import numpy as np
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from copy import deepcopy
import networkx as nx
import os
import argparse
from torch import nn
from torch.nn import functional as F
import torch_geometric as tg
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx, from_networkx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from src.plot.graph import graph_img
import warnings
warnings.filterwarnings("ignore")
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def chebyshev_distance(p, q):
    """
    Calculate the Chebyshev distance between two vectors
    
    Parameters:
    p: The first vector, format (p1, p2, ..., pn)
    q: The second vector, format (q1, q2, ..., qn)
    
    Returns:
    The Chebyshev distance
    """
    # Ensure vectors have the same length
    assert len(p) == len(q), "Vectors must have the same length"
    
    # Calculate the maximum absolute difference of each coordinate
    max_diff = max(abs(p_i - q_i) for p_i, q_i in zip(p, q))
    
    return max_diff

def total_distance(p1, p2, categorical_var, numerical_var):
    """
    Calculate the total distance between two patients
    
    Parameters:
    p1: The first patient's diagnosis, format (p1, p2, ..., pn)
    p2: The second patient's diagnosis, format (q1, q2, ..., qn)
    
    Returns:
    The total distance
    """
    # Calculate the total distance
    # print(p1)
    # print(p2)
    distance = 0
    for var in categorical_var:
        # make the set for p1 and p2
        set1 = set([c for c in categorical_var[var] if p1[c] == 1])
        set2 = set([c for c in categorical_var[var] if p2[c] == 1])
        distance += 1 - jaccard_similarity(set1, set2)
    distance /= len(categorical_var)
    p1_var = tuple(p1[var] for var in numerical_var)
    p2_var = tuple(p2[var] for var in numerical_var)
    distance += chebyshev_distance(p1_var, p2_var)
    distance /= 2
    return distance

def variables_collect(df):
    categorical_variables = {}
    numerical_variables = []
    columns = df.columns.tolist()
    for col in columns:
        # if the column only contains 0 and 1, it is a categorical variable
        if '_' in col and len(set(df[col])) == 2:
            prefix = col.split('_')[0]
            if prefix not in categorical_variables:
                # find all the columns with the same prefix
                related_columns = [c for c in columns if c.startswith(prefix)]
                categorical_variables[prefix] = related_columns
        if '_' not in col and len(set(df[col])) > 2:
            numerical_variables.append(col)
    return categorical_variables, numerical_variables

def cut_diff_class(data):
    '''
    Cutting the edges into different classes

    Args:
        torch_geometric.data.Data: the graph data
    Returns:
        torch_geometric.data.Data: the graph data with edge class
    '''
    edge_index = data.edge_index
    y = data.y
    edge_class = y[edge_index[0]] != y[edge_index[1]]
    # delete the edge in different class
    edge_index = edge_index[:, ~edge_class]
    data.edge_index = edge_index
    return data

def graph_mst(data):
    '''
    Get the minimum spanning tree of the graph

    Args:
        torch_geometric.data.Data: the graph data

    Returns:
        torch_geometric.data.Data: the graph data with mst
    '''
    # Convert to NetworkX graph for minimum spanning tree calculation
    G = to_networkx(data, to_undirected=True)
    # Convert NetworkX graph to adjacency matrix
    adj_matrix = nx.adjacency_matrix(G)
    # Convert adjacency matrix to CSR format and ensure data type
    adj_matrix = csr_matrix(adj_matrix, dtype=np.float64)
    # Compute the minimum spanning tree
    mst = minimum_spanning_tree(adj_matrix)
    # Construct NetworkX graph from the MST sparse matrix
    mst_graph = nx.Graph()
    for i, j in zip(*mst.nonzero()):
        mst_graph.add_edge(i, j)
    # Convert NetworkX graph to PyTorch Geometric format
    edge_index = torch.tensor(list(mst_graph.edges()), dtype=torch.long).t().contiguous()
    # Update the data object with the MST edge index
    data.edge_index = edge_index
    return data

class ADPGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, log=False, raw_dir=None, processed_dir=None, day=None, cutoff=None, buffer=None, random_state=None, test_train_split_part=None, task=None, df=None, graph=None, threshold=None, img_path=None, img_name=None, cfg=None, **kwargs):
    # def __init__(self, root, transform=None, pre_transform=None, log=False):
        # super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        # self.raw_dir = raw_dir
        # self.processed_dir = processed_dir
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.day = day
        self.cutoff = cutoff
        self.buffer = buffer
        self.random_state = random_state
        self.test_train_split_part = test_train_split_part
        self.task = task
        self.df = df
        self.graph = graph
        self.threshold = threshold
        self.img_path = img_path
        self.img_name = img_name
        self.cfg = cfg
        if 'mst' in self.task and 'sep' not in self.task:
            self.graph = '_'.join(self.graph.split('_')[:3]) + '_mst_'+ self.cfg['run_config']['task'] + '.' + self.graph.split('.')[1]
        elif 'sep' in self.task and 'mst' not in self.task:
            self.graph = '_'.join(self.graph.split('_')[:3]) + '_sep_'+ self.cfg['run_config']['task'] + '.' + self.graph.split('.')[1]
        elif 'mst_sep' in self.task:
            self.graph = '_'.join(self.graph.split('_')[:3]) + '_mst_sep_'+ self.cfg['run_config']['task'] + '.' + self.graph.split('.')[1]
        elif 'sep_mst' in self.task:
            self.graph = '_'.join(self.graph.split('_')[:3]) + '_sep_mst_'+ self.cfg['run_config']['task'] + '.' + self.graph.split('.')[1]
        else:
            self.graph = self.graph
        super(ADPGraphDataset, self).__init__(root, transform=transform, pre_transform=pre_transform, log=log)
        if self.cfg['run_config']['task'] == 'multignn' or self.cfg['run_config']['task'] == 'graph_classification':
            self.data_list = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        # return [self.df]
        return [self.df] # or 'mci_ad.tsv', ad_death.tsv
    
    @property
    def processed_file_names(self)  -> Union[str, List[str], Tuple[str, ...]]:
        # return [self.graph]
        return [self.graph] # or 'graph_mci_ad.gml', graph_ad_death.gml
 
    def download(self):
        pass
    
    def process(self):
        data_list = []
        self.data_name = self.raw_file_names[0]
        self.data_path = os.path.join(self.raw_dir, self.data_name)
        df = pd.read_csv(self.data_path, sep='\t')
        df = self.label_generator(df)
        graph = self.graph_generator_node(df)
        data = self.graph_org_node(graph, df)
        data = data if self.pre_transform is None else self.pre_transform(data)
        graph_img(data, self.img_path, self.img_name)
        data['task'] == 'node'
        data_list.append(data)
        if self.cfg['run_config']['task'] == 'multignn' or self.cfg['run_config']['task'] == 'graph_classification':
            # replicate the graph for the multi-gnn model
            for _ in range(data.x.size(0) - 1):
                data_list.append(deepcopy(data))
            graph_data_list = self.graph_generator_graph(df)
            for i, graph in enumerate(graph_data_list):
                data = self.graph_org_graph(graph, df, i)
                data = data if self.pre_transform is None else self.pre_transform(data)
                data['task'] == 'graph'
                data_list.append(data)
        else:
            data, slices = self.collate(data_list)
        if self.buffer:
            if self.cfg['run_config']['task'] == 'binary_classification':
                # dataset_name = '_'.join(self.graph.split('_')[:3]) + '_' + str(self.buffer) + '_' + str(round(self.buffer_day, 4)) + '_' + str(self.cutoff) + '_binary_classification.dataset'
                dataset_name = '_'.join(self.graph.split('_')[:3]) + '_' + str(self.buffer) + '_' + str(round(self.buffer_day, 4)) + '_' + str(self.cutoff) + '_binary_classification.dataset'
            elif self.cfg['run_config']['task'] == 'multi_classification':
                dataset_name = '_'.join(self.graph.split('_')[:3]) + '_' + str(self.buffer) + '_' + str(round(self.buffer_day, 4)) + '_' + str(self.cutoff) + '_multi_classification.dataset'
            elif self.cfg['run_config']['task'] == 'regression':
                dataset_name = '_'.join(self.graph.split('_')[:3]) + '_' + str(self.buffer) + '_' + str(round(self.buffer_day, 4)) + '_' + str(self.cutoff) + '_regression.dataset'
            elif self.cfg['run_config']['task'] == 'graph_classification':
                dataset_name = '_'.join(self.graph.split('_')[:3]) + '_' + str(self.buffer) + '_' + str(round(self.buffer_day, 4)) + '_' + str(self.cutoff) + '_graph_classification.dataset'
            elif self.cfg['run_config']['task'] == 'multignn':
                dataset_name = '_'.join(self.graph.split('_')[:3]) + '_' + str(self.buffer) + '_' + str(round(self.buffer_day, 4)) + '_' + str(self.cutoff) + '_multignn.dataset'
            else:
                raise ValueError(f'Invalid task: {self.task}')
        else:
            if self.cfg['run_config']['task'] == 'binary_classification':
                dataset_name = '_'.join(self.graph.split('_')[:3]) + '_' + str(self.cutoff) + '_binary_classification.dataset'
            elif self.cfg['run_config']['task'] == 'multi_classification':
                dataset_name = '_'.join(self.graph.split('_')[:3]) + '_' + str(self.cutoff) + '_multi_classification.dataset'
            elif self.cfg['run_config']['task'] == 'regression':
                dataset_name = '_'.join(self.graph.split('_')[:3]) + '_' + str(self.cutoff) + '_regression.dataset'
            elif self.cfg['run_config']['task'] == 'graph_classification':
                dataset_name = '_'.join(self.graph.split('_')[:3]) + '_' + str(self.cutoff) + '_graph_classification.dataset'
            elif self.cfg['run_config']['task'] == 'multignn':
                dataset_name = '_'.join(self.graph.split('_')[:3]) + '_' + str(self.cutoff) + '_multignn.dataset'
            else:
                raise ValueError(f'Invalid task: {self.task}')
        if self.cfg['run_config']['task'] == 'multignn' or self.cfg['run_config']['task'] == 'graph_classification':
            torch.save(data_list, self.processed_paths[0])
        else:
            torch.save((data, slices), os.path.join(os.path.dirname(self.processed_paths[0]), dataset_name))
            

    def label_generator(self, df):
        '''
        Deal with the raw data, generate the label for each patient by the cutoff day and buffer
        '''
        df_copy = df.copy()
        # sort by day
        df_copy = df_copy.sort_values(by=self.day, ascending=True)
        
        def cutoff_percent_to_day(cutoff):
            min_day = df_copy[self.day].min()
            max_day = df_copy[self.day].max()
            mean_day = df_copy[self.day].mean()
            # calculate the cutoff percentile
            # cutoff_day = np.percentile(df_copy[day], cutoff*100)
            cutoff_day = cutoff
            print(f'the cutoff is {cutoff}')
            print(f'the min day is {min_day}')
            print(f'the max day is {max_day}')
            print(f'the mean day is {mean_day}')
            # print(f'the cutoff day is {cutoff_day}')
            return cutoff_day

        cutoff_day = cutoff_percent_to_day(self.cutoff)

        def delete_buffer():
            # delete buffer/2 before and after cutoff
            self.buffer_day = np.percentile(df_copy[self.day], self.buffer*100)
            buffer_day_bisect = int(self.buffer_day / 2)
            buffer_init_day = cutoff_day - buffer_day_bisect
            buffer_end_day = cutoff_day + buffer_day_bisect
            buffer_init = df_copy[self.day].sub(buffer_init_day).abs().idxmin()
            buffer_end = df_copy[self.day].sub(buffer_end_day).abs().idxmin()
            if buffer_init == buffer_end and buffer_end < len(df_copy) - 1:
                buffer_end = buffer_end + 1
            # find cutoff index which is the most close to cutoff
            cutoff_index = df_copy[self.day].sub(cutoff_day).abs().idxmin()
            # buffer_num = int(len(df) * buffer / 2)
            # buffer_init = cutoff_index - buffer_num
            # buffer_end = cutoff_index + buffer_num
            print(f'buffer_day is {self.buffer_day}')
            print(f'buffer_day_bisect is {buffer_day_bisect}')
            print(f'buffer_init is {buffer_init}')
            print(f'buffer_init_day is {buffer_init_day}')
            print(f'buffer_end is {buffer_end}')
            print(f'buffer_end_day is {buffer_end_day}')
            print(f'cutoff_index is {cutoff_index}')
            return buffer_init, buffer_end

        if self.buffer:
            buffer_init, buffer_end = delete_buffer()
            # delete buffer
            df_copy = df_copy.drop(df_copy.index[buffer_init:buffer_end])
        
        df_copy['label'] = (df_copy[self.day] < cutoff_day).astype(int)
        # reset the index
        df_copy = df_copy.reset_index(drop=True)
        # save df_copy
        if self.buffer:
            df_new_name = '_'.join(self.graph.split('_')[:3]) + '_' + str(self.buffer) + '_' + str(round(self.buffer_day, 4)) + '_' + str(cutoff_day) + '.tsv'
        else:
            df_new_name = '_'.join(self.graph.split('_')[:3]) + '_' + str(cutoff_day) + '.tsv'
        df_copy.to_csv(os.path.join(self.root, 'raw', df_new_name), sep='\t', index=False)
        return df_copy
    
    def graph_generator_node(self, df, **kwargs):
        df_copy = df.copy()
        node = df_copy['Participant ID']
        label = df_copy[self.day] # ad_death Days Between AD and Death
        # label = df_copy['label'] # mci_ad Time Between MCI and AD
        # days = df_copy[self.day]
        # label = df['Time Between MCI and AD'] # mci_ad Time Between MCI and AD
        # df_copy = df_copy.drop(['Participant ID', days, 'label'], axis=1)
        df_copy = df_copy.drop(['Participant ID', self.day], axis=1)
        categorical_var, numerical_var = variables_collect(df_copy)
        # df_copy = pd.concat([node, label, days, df_copy], axis=1)
        df_copy = pd.concat([node, label, df_copy], axis=1)
        # initialize a new graph
        G = nx.Graph()
        # G.add_nodes_from(patients_diagnosis.keys())
        # G.add_nodes_from(df.index)
        G.add_nodes_from(df_copy['Participant ID'])


        # initialize the threshold
        threshold = self.threshold # MCI to AD: over 0.7, so we try 0.6 to 0.7 with the step 0.01, nearly 0.68, # AD_Death: over 0.7, so we try 0.7 to 0.8
        # for p1, p2 in combinations(range(df.shape[0]), 2):
        #     # print(df.iloc[p1], df.iloc[p2], categorical_var, numerical_var)
        #     print(total_distance(df.iloc[p1], df.iloc[p2], categorical_var, numerical_var))
        #     # break

        # Gradually lower the threshold until obtain a connected graph
        count = 0
        while True:
            edge_index = []  # Reinitialize edge_index
            # for patient1, patient2 in combinations(range(df.shape[0]), 2):
            for patient1, patient2 in tqdm(combinations(df_copy['Participant ID'], 2), total=len(df_copy) * (len(df_copy) - 1) // 2):
                # Calculate distance between patients
                # similarity = jaccard_similarity(patients_diagnosis[patient1], patients_diagnosis[patient2])
                # our df
                distance = total_distance(df_copy[df_copy['Participant ID'] == patient1].iloc[0], df_copy[df_copy['Participant ID'] == patient2].iloc[0], categorical_var, numerical_var)
                # # toy df
                # distance = total_distance(df.iloc[patient1], df.iloc[patient2], categorical_var, numerical_var)
                if distance <= threshold:
                    edge_index.append((patient1, patient2))
            if len(edge_index) > 0:
                G.add_edges_from(edge_index)
                if nx.is_connected(G):
                    # # draw the graph
                    # plt.clf()
                    # nx.draw(G, with_labels=True, font_weight='bold')
                    # plt.savefig(f"/home/weimin.meng/projects/AD_progression/img/graphs/graph{count}.png")
                    # # output
                    print("The constructed edge_index：", edge_index)
                    print("The final threshold: ", threshold)
                    return G
                # else:
                #     plt.clf()
                #     nx.draw(G, with_labels=True, font_weight='bold')
                #     plt.savefig(f"/home/weimin.meng/projects/AD_progression/img/graphs/graph{count}.png")
                #     # count += 1
            threshold += 0.01 # 0.1 * 0.1^n
            count += 1
            print('the current threshold:', threshold)
            print('the current round number:', count)
            if threshold > 1:
                print('The threshold is too high, the graph is not connected')
                break
        return None
    
    # def graph_generator_graph(self, df, **kwargs):
    #     df_copy = df.copy()
    #     # node = df_copy['Participant ID']
    #     # # label = df_copy[self.day] # ad_death Days Between AD and Death
    #     # label = df_copy['label'] # mci_ad Time Between MCI and AD
    #     # days = df_copy[self.day]
    #     df_copy = df_copy.drop(['Participant ID', self.day, 'label'], axis=1)
    #     # initialize a new graph
    #     column_index_map = {col: idx for idx, col in enumerate(df_copy.columns)}
    #     graph_list = []
    #     for index, row in tqdm(df_copy.iterrows(), total=len(df_copy)):
    #         # initialize the threshold
    #         threshold = 1 # MCI to AD: over 0.7, so we try 0.6 to 0.7 with the step 0.01, nearly 0.68, # AD_Death: over 0.7, so we try 0.7 to 0.8
    #         G = nx.Graph()
    #         G.add_nodes_from(column_index_map.values())

    #         # Gradually lower the threshold until obtain a connected graph
    #         count = 0
    #         while True:
    #             edge_index = []  # Reinitialize edge_index
    #             # for patient1, patient2 in combinations(range(df.shape[0]), 2):
    #             for feature1, feature2 in tqdm(combinations(df_copy.columns, 2), total=len(df_copy.columns) * (len(df_copy.columns) - 1) // 2):
    #                 # Calculate distance between patients
    #                 # similarity = jaccard_similarity(patients_diagnosis[patient1], patients_diagnosis[patient2])
    #                 # our df
    #                 value1 = row[feature1]
    #                 value2 = row[feature2]
    #                 vector1 = np.array([value1]).reshape(1, -1)
    #                 vector2 = np.array([value2]).reshape(1, -1)
    #                 similarity = abs(cosine_similarity(vector1, vector2)[0][0])
    #                 # # toy df
    #                 # distance = total_distance(df.iloc[patient1], df.iloc[patient2], categorical_var, numerical_var)
    #                 if similarity >= threshold:
    #                     edge_index.append((column_index_map[feature1], column_index_map[feature2]))
    #             if len(edge_index) > 0:
    #                 G.add_edges_from(edge_index)
    #                 if nx.is_connected(G):
    #                     # # draw the graph
    #                     # plt.clf()
    #                     # nx.draw(G, with_labels=True, font_weight='bold')
    #                     # plt.savefig(f"/home/weimin.meng/projects/AD_progression/img/graphs/graph{count}.png")
    #                     # # output
    #                     print("The constructed edge_index：", edge_index)
    #                     print("The final threshold: ", threshold)
    #                     break
    #                 # else:
    #                 #     plt.clf()
    #                 #     nx.draw(G, with_labels=True, font_weight='bold')
    #                 #     plt.savefig(f"/home/weimin.meng/projects/AD_progression/img/graphs/graph{count}.png")
    #                 #     # count += 1
    #             threshold -= 0.1 # 0.1 * 0.1^n
    #             count += 1
    #             print('the current threshold:', threshold)
    #             print('the current round number:', count)
    #             if threshold < 0:
    #                 raise ValueError('The threshold is too high, the graph is not connected')
    #         graph_list.append(G)
    #     if len(graph_list) != len(df_copy):
    #         raise ValueError('The graph number is not equal to the dataframe row number')
    #     return graph_list

    def graph_generator_graph(self, df, graph_type='erdos_renyi', **kwargs):
        # Copy DataFrame
        df_copy = df.copy()

        # Create a mapping of column names to their indices
        column_index_map = {col: idx for idx, col in enumerate(df_copy.columns)}

        # Initialize a list to store generated graphs
        graph_list = []

        # Iterate over rows in the DataFrame
        for index, row in tqdm(df_copy.iterrows(), total=len(df_copy), desc='Generating graphs'):
            # Initialize an empty graph
            G = nx.Graph()

            # Generate different types of graphs based on the specified type
            if graph_type == 'erdos_renyi':
                # Generate Erdős-Rényi graph
                G = nx.erdos_renyi_graph(len(df_copy.columns)-3, 0.1) 
            elif graph_type == 'connected_watts_strogatz':
                # Generate connected Watts-Strogatz graph
                G = nx.connected_watts_strogatz_graph(len(df_copy.columns)-3, 4, 0.3) 
            elif graph_type == 'barabasi_albert':
                # Generate Barabási-Albert graph
                G = nx.barabasi_albert_graph(len(df_copy.columns)-3, 3) 
            else:
                raise ValueError(f'Invalid graph type: {graph_type}')

            # Append the generated graph to the list
            graph_list.append(G)

        return graph_list
    
    def graph_org_node(self, graph, df, **kwargs):
        '''
        generate the graph data after the algorithm from networkx graph to torch_geometric graph

        Args:
            networkx graph: the graph data
            df: the dataframe of the dataframe data

        Returns:
            torch_geometric.data.Data: the graph data
        '''
        new_column_order = ["Participant ID", self.day, "label"] + [col for col in df.columns if col not in ["Participant ID", self.day, "label"]]
        df = df.reindex(columns=new_column_order)
        id_to_index = {participant_id: i for i, participant_id in enumerate(df["Participant ID"])}
        edges_index = [(id_to_index[int(source)], id_to_index[int(target)]) for source, target in graph.edges()]

        # X = df_mci_ad.drop(columns=["Participant ID", "Time Between MCI and AD", "label"]).values
        if self.cfg['run_config']['task'] == 'binary_classification' or self.cfg['run_config']['task'] == 'graph_classification' or self.cfg['run_config']['task'] == 'multignn':
            y = df["label"].values
        elif self.cfg['run_config']['task'] == 'multi_classification':
            y = df['label'].values
        elif self.cfg['run_config']['task'] == 'regression':
            y = df[self.day].values
        else:
            raise ValueError(f'Invalid task: {self.cfg["run_config"]["task"]}')

        X_train, X_temp, y_train, y_temp = train_test_split(df, y, test_size=self.test_train_split_part, random_state=self.random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=self.random_state)

        num_samples = len(df)
        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        val_mask = torch.zeros(num_samples, dtype=torch.bool)
        test_mask = torch.zeros(num_samples, dtype=torch.bool)

        train_indices = df.index[df["Participant ID"].isin(X_train["Participant ID"])].tolist()
        val_indices = df.index[df["Participant ID"].isin(X_val["Participant ID"])].tolist()
        test_indices = df.index[df["Participant ID"].isin(X_test["Participant ID"])].tolist()

        train_mask[train_indices] = 1
        val_mask[val_indices] = 1
        test_mask[test_indices] = 1

        data = Data(x=torch.tensor(df.drop(columns=["Participant ID", self.day, "label"]).values, dtype=torch.float),
                    edge_index=torch.tensor(edges_index, dtype=torch.long).t().contiguous(),
                    y=torch.tensor(y, dtype=torch.long),
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask, task='node')

        # if self.task == 'all':
        #     return data
        if 'mst' in self.task and 'sep' not in self.task:
            return graph_mst(data)
        elif 'sep' in self.task and 'mst' not in self.task:
            return cut_diff_class(data) # cut the edges into different classes
        elif 'mst_sep' in self.task:
            data = graph_mst(data)
            data = cut_diff_class(data)
            return data
        elif 'sep_mst' in self.task:
            data = cut_diff_class(data)
            data = graph_mst(data)
            return data
        else:
            return data
            # raise ValueError(f'Invalid task: {self.task}')
        
    def graph_org_graph(self, graph, df, row_idx, **kwargs):
        '''
        generate the graph data after the algorithm from networkx graph to torch_geometric graph

        Args:
            networkx graph: the graph data
            df: the dataframe of the dataframe data

        Returns:
            torch_geometric.data.Data: the graph data
        '''
        new_column_order = ["Participant ID", self.day, "label"] + [col for col in df.columns if col not in ["Participant ID", self.day, "label"]]
        df = df.reindex(columns=new_column_order)
        row = df.iloc[row_idx]
        # id_to_index = {participant_id: i for i, participant_id in enumerate(df["Participant ID"])}
        # edges_index = [(id_to_index[int(source)], id_to_index[int(target)]) for source, target in graph.edges()]

        # X = df_mci_ad.drop(columns=["Participant ID", "Time Between MCI and AD", "label"]).values
        if self.cfg['run_config']['task'] == 'binary_classification' or self.cfg['run_config']['task'] == 'graph_classification' or self.cfg['run_config']['task'] == 'multignn':
            y = np.array([df.loc[row_idx, "label"]])
            # replicate y to the same size of the graph
        elif self.cfg['run_config']['task'] == 'multi_classification':
            y = np.array([df.loc[row_idx, "label"]])
        elif self.cfg['run_config']['task'] == 'regression':
            y = np.array([df.loc[row_idx, self.day]])
        else:
            raise ValueError(f'Invalid task: {self.cfg["run_config"]["task"]}')


        # X_train, X_temp, y_train, y_temp = train_test_split(row, y, test_size=self.test_train_split_part, random_state=self.random_state)
        # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=self.random_state)

        num_samples = len(df)
        # train_mask = torch.zeros(num_samples, dtype=torch.bool)
        # val_mask = torch.zeros(num_samples, dtype=torch.bool)
        # test_mask = torch.zeros(num_samples, dtype=torch.bool)

        # train_indices = row.index[row["Participant ID"].isin(X_train["Participant ID"])].tolist()
        # val_indices = row.index[row["Participant ID"].isin(X_val["Participant ID"])].tolist()
        # test_indices = row.index[row["Participant ID"].isin(X_test["Participant ID"])].tolist()

        # train_mask[train_indices] = 1
        # val_mask[val_indices] = 1
        # test_mask[test_indices] = 1
        x = torch.tensor(row.drop(["Participant ID", "Time Between MCI and AD", "label"], axis=0).values, dtype=torch.float)
        x = x.expand(x.shape[0], -1)
        x = x.T
        y = torch.tensor(y, dtype=torch.long)
        y = y.repeat(num_samples)
        train_mask = torch.ones(x.shape[0], dtype=torch.bool)
        val_mask = torch.zeros(x.shape[0], dtype=torch.bool)
        test_mask = torch.zeros(x.shape[0], dtype=torch.bool)
        data = from_networkx(graph)
        data = Data(x=x,
                    edge_index=data.edge_index,
                    y=y,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask,
                    task='graph')
        data = graph_mst(data)

        return data