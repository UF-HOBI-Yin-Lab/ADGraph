import networkx as nx
import torch
import torch.optim as optim
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from src.utils.seed import set_random_seed
from src.model.metrics import BinaryClassificationMetrics, RegressionMetrics
from src.model.models import MLP, GCN, GAT, SAGE
from src.model.baseline import logistic_regression, random_forest, xgboost, BiLSTM, ResNet
from src.model.data import ADPGraphDataset
from sklearn.model_selection import ShuffleSplit
from src.model.MultiGNN import MultiGNN, GraphClassifier, NodeClassifier
import pandas as pd
import os.path as osp
import pickle
import os
import time
import json
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

class Runner():
    '''
    Train class to train the model
    Variables:
        - cfg: Configuration file
        - retrain: Retrain flag
        - device: Device
        - dl: Download flag
        - task: Task
        - training: Training
        - project: Project
        - loss: Loss
        - accuracy: Accuracy
        - precision: Precision
        - recall: Recall
        - f1: F1
        - auroc: AUROC
        - mse: Mean Squared Error
        - mae: Mean Absolute Error
        - r2: R-squared
        - metrics: Metrics
        - df: Dataframe
        - graph: Graph
        - dataset: Train dataset
        - dataloader: Train loader
        - input_dim: Input dimension
        - output_dim: Output dimension
        - model: Model
        - optimizer: Optimizer
        - scheduler: Scheduler

    '''
    def __init__(self, cfg, **kwargs):
        '''
        load variables
        '''
        self.cfg = cfg # Read config from args
        self.retrain = self.cfg['run_config']['retrain'] # Retrain flag
        self.dl = self.cfg['run_config']['dl'] # Download flag
        self.task = self.cfg['run_config']['task']
        self.training = self.cfg['run_config']['training']
        self.device = self.initial_device()
        '''
        Initialize seed, loss, metrics, model, dataset, optimizer
        '''
        if 'mci_ad' in self.cfg['run_config']['dataset']: # Read the metrics
            self.project = 'mci_ad'
        elif 'ad_death' in self.cfg['run_config']['dataset']:
            self.project = 'ad_death'
        else:
            raise ValueError(f'Unsupported task: {self.cfg["run_config"]["dataset"]}')
        if self.cfg['run_config']['seed']:
            model_config = self.project + '_config'
            set_random_seed(self.cfg[model_config]['optimal_random_state'])  # Initialize random seed from utils.seed.py using set_random_seed
        self.clear_metrics() # Clear and initialize loss value
        if self.task == 'binary_classification' or self.task == 'graph_classification' or self.task == 'multignn': # Read the metrics
            self.metrics = BinaryClassificationMetrics()
        elif self.task == 'multi_classification':
            pass
        elif self.task == 'regression':
            self.metrics = RegressionMetrics()
        else:
            raise ValueError(f'Unsupported task: {self.task}')

        # self.dataset = {} # Read train dataset from args, as like the dict {'mci_ad': {'df': df, 'graph': graph, 'dataset': adpgraph}, 'ad_death': {'df': df, 'graph': graph, 'dataset': adpgraph}}
        # for dataset_name in self.cfg['run_config']['datasets']:
        #     self.dataset[dataset_name] = self.get_dataset(dataset_name)
        self.df, self.graph, self.dataset = self.get_dataset(self.cfg['run_config']['dataset'])
        if self.cfg['run_config']['dataloader']:
            self.dataloader = self.get_dataloader()
        self.input_dim, self.output_dim = self.get_in_out_dim()
        if self.cfg['run_config']['task'] == 'graph_classification':
            self.output_dim = 1
        # self.models = {} 
        # for model_name in self.cfg['run_config']['models']:
        #     self.models_mci_ad.append(self.get_model(model_name, self.cfg['mci_ad_config']['optimal_cv_n_split'], self.cfg['mci_ad_config']['optimal_random_state'], self.cfg['mci_ad_config']['optimal_tt_split_part']))
        if self.project == 'mci_ad':
            if self.cfg['run_config']['task'] == 'graph_classification' or self.cfg['run_config']['task'] == 'multignn':
                self.model = self.get_model(self.cfg['run_config']['task'], self.cfg['mci_ad_config']['optimal_cv_n_split'], self.cfg['mci_ad_config']['optimal_random_state'], self.cfg['mci_ad_config']['optimal_tt_split_part'], self.cfg['run_config']['model'])
            else:
                self.model = self.get_model(self.cfg['run_config']['model'], self.cfg['mci_ad_config']['optimal_cv_n_split'], self.cfg['mci_ad_config']['optimal_random_state'], self.cfg['mci_ad_config']['optimal_tt_split_part'], self.cfg['run_config']['model'])
        else:
            if self.cfg['run_config']['task'] == 'graph_classification' or self.cfg['run_config']['task'] == 'multignn':
                self.model = self.get_model(self.cfg['run_config']['task'], self.cfg['ad_death_config']['optimal_cv_n_split'], self.cfg['ad_death_config']['optimal_random_state'], self.cfg['ad_death_config']['optimal_tt_split_part'], self.cfg['run_config']['model'])
            else:
                self.model = self.get_model(self.cfg['run_config']['model'], self.cfg['ad_death_config']['optimal_cv_n_split'], self.cfg['ad_death_config']['optimal_random_state'], self.cfg['ad_death_config']['optimal_tt_split_part'], self.cfg['run_config']['model'])
        self.optimizer = self.optimizer(self.model, self.cfg['run_config']['weight_decay'])
        self.scheduler = self.scheduler()

    def initial_device(self):
        # Set device
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{self.cfg["run_config"]["cuda"]}')
        else:
            device = torch.device('cpu')
        return device
    
    def clear_metrics(self):
        if self.cfg['run_config']['task'] == 'binary_classification' or self.cfg['run_config']['task'] == 'multi_classification' or self.cfg['run_config']['task'] == 'graph_classification' or self.cfg['run_config']['task'] == 'multignn':
            self.loss = 0.0  # Clear and initialize loss value
            self.accuracy = 0.0
            self.precision = 0.0
            self.recall = 0.0
            self.f1 = 0.0
            self.auroc = 0.0
        elif self.cfg['run_config']['task'] == 'regression':
            self.loss = 0.0
            self.mse = 0.0
            self.mae = 0.0
            self.r2 = 0.0
        else:
            raise ValueError(f'Unsupported task: {self.cfg["run_config"]["task"]}')

    def get_in_out_dim(self):
        # Get input and output dimensions
        # input_dim = len(self.dataset[0][0])
        if hasattr(self.dataset, 'data_list'):
            input_dim = self.dataset.data_list[0].num_node_features
            output_dim = len(self.dataset.data_list[0].y)
        else:
            input_dim = self.dataset.num_node_features
            # output_dim = len(set(self.dataset[:][1]))  # Assuming labels are at the second position
            output_dim = len(self.dataset.y.shape)
        return input_dim, output_dim
    
    def get_dataset(self, dataset_name):
        if dataset_name == 'mci_ad':
            df = pd.read_csv(self.cfg['mci_ad_config']['path'], sep='\t')
            if self.cfg['run_config']['graph_data']:
                graph = nx.read_gml(self.cfg['mci_ad_config']['graph_path'])
            else:
                graph = None
        elif dataset_name == 'mci_ad_mst':
            df = pd.read_csv(self.cfg['mci_ad_config']['path'], sep='\t')
            if self.cfg['run_config']['graph_data']:
                graph = nx.read_gml(self.cfg['mci_ad_config']['graph_path_mst'])
            else:
                graph = None
        elif dataset_name == 'mci_ad_sep':
            df = pd.read_csv(self.cfg['mci_ad_config']['path'], sep='\t')
            if self.cfg['run_config']['graph_data']:
                graph = nx.read_gml(self.cfg['mci_ad_config']['graph_path_sep'])
            else:
                graph = None
        elif dataset_name == 'mci_ad_mst_sep':
            df = pd.read_csv(self.cfg['mci_ad_config']['path'], sep='\t')
            if self.cfg['run_config']['graph_data']:
                graph = nx.read_gml(self.cfg['mci_ad_config']['graph_path_mst_sep'])
            else:
                graph = None
        elif dataset_name == 'mci_ad_sep_mst':
            df = pd.read_csv(self.cfg['mci_ad_config']['path'], sep='\t')
            if self.cfg['run_config']['graph_data']:
                graph = nx.read_gml(self.cfg['mci_ad_config']['graph_path_sep_mst'])
            else:
                graph = None
        elif dataset_name == 'ad_death':
            df = pd.read_csv(self.cfg['ad_death_config']['path'], sep='\t')
            if self.cfg['run_config']['graph_data']:
                graph = nx.read_gml(self.cfg['ad_death_config']['graph_path'])
            else:
                graph = None
        elif dataset_name == 'ad_death_mst':
            df = pd.read_csv(self.cfg['ad_death_config']['path'], sep='\t')
            if self.cfg['run_config']['graph_data']:
                graph = nx.read_gml(self.cfg['ad_death_config']['graph_path_mst'])
            else:
                graph = None
        elif dataset_name == 'ad_death_sep':
            df = pd.read_csv(self.cfg['ad_death_config']['path'], sep='\t')
            if self.cfg['run_config']['graph_data']:
                graph = nx.read_gml(self.cfg['ad_death_config']['graph_path_sep'])
            else:
                graph = None
        elif dataset_name == 'ad_death_mst_sep':
            df = pd.read_csv(self.cfg['ad_death_config']['path'], sep='\t')
            if self.cfg['run_config']['graph_data']:
                graph = nx.read_gml(self.cfg['ad_death_config']['graph_path_mst_sep'])
            else:
                graph = None
        elif dataset_name == 'ad_death_sep_mst':
            df = pd.read_csv(self.cfg['ad_death_config']['path'], sep='\t')
            if self.cfg['run_config']['graph_data']:
                graph = nx.read_gml(self.cfg['ad_death_config']['graph_path_sep_mst'])
            else:
                graph = None
        else:
            raise ValueError(f'Unsupported dataset: {dataset_name}')

        if 'mci_ad' in dataset_name:
            adpgraph = ADPGraphDataset(root = self.cfg['mci_ad_config']['root'],# root=os.path.dirname(self.cfg['mci_ad_config']['root']),
                                       transform=None, pre_transform=None, log=None, raw_dir=os.path.dirname(self.cfg['mci_ad_config']['path']), processed_dir=os.path.dirname(self.cfg['mci_ad_config']['graph_path']), day=self.cfg['mci_ad_config']['day'],
                                       cutoff=self.cfg['mci_ad_config']['optimal_cutoff'], buffer=self.cfg['mci_ad_config']['optimal_buffer'], 
                                       random_state=self.cfg['mci_ad_config']['optimal_random_state'], 
                                       test_train_split_part=self.cfg['mci_ad_config']['optimal_tt_split_part'], task=dataset_name,
                                       df=os.path.basename(self.cfg['mci_ad_config']['path']), graph=os.path.basename(self.cfg['mci_ad_config']['graph_buffer']), threshold=self.cfg['mci_ad_config']['threshold'], img_path=self.cfg['mci_ad_config']['img_path'], img_name=self.cfg['run_config']['dataset'], cfg=self.cfg)
            # adpgraph = ADPGraphDataset(root=None, transform=None, pre_transform=None, log=None)
        elif 'ad_death' in dataset_name:
            adpgraph = ADPGraphDataset(root = self.cfg['ad_death_config']['root'], # root=os.path.dirname(self.cfg['ad_death_config']['graph_path']),
                                       transform=None, pre_transform=None, log=None, raw_dir=os.path.dirname(self.cfg['ad_death_config']['path']), processed_dir=os.path.dirname(self.cfg['ad_death_config']['graph_path']), day=self.cfg['ad_death_config']['day'],
                                       cutoff=self.cfg['ad_death_config']['optimal_cutoff'], buffer=self.cfg['ad_death_config']['optimal_buffer'], 
                                       random_state=self.cfg['ad_death_config']['optimal_random_state'], 
                                       test_train_split_part=self.cfg['ad_death_config']['optimal_tt_split_part'], task=dataset_name,
                                       df=os.path.basename(self.cfg['ad_death_config']['path']), graph=os.path.basename(self.cfg['ad_death_config']['graph_buffer']), threshold=self.cfg['ad_death_config']['threshold'], img_path=self.cfg['ad_death_config']['img_path'], img_name=self.cfg['run_config']['dataset'], cfg=self.cfg)

        return df, graph, adpgraph
    
    def get_dataloader(self):
        # Return DataLoader for the training dataset
        dataloader = DataLoader(self.dataset, batch_size=self.cfg['run_config']['batch_size'], shuffle=self.cfg['run_config']['shuffle'], num_workers=self.cfg['run_config']['num_workers'])
        return dataloader

    def load_model(self, model_path, model):
        if self.dl:
            # Construct the model file paths, considering both ".pt" and ".pth" suffixes
            model_file_pt = os.path.join(model_path, model + ".pt")
            model_file_pth = os.path.join(model_path, model + ".pth")

            # Check if the model file with ".pt" suffix exists
            if os.path.exists(model_file_pt):
                # If the file exists, load the model
                model = torch.load(model_file_pt)
                return model
            # Check if the model file with ".pth" suffix exists
            elif os.path.exists(model_file_pth):
                # If the file exists, load the model
                model = torch.load(model_file_pth)
                return model
            else:
                # If neither ".pt" nor ".pth" model file exists, raise an exception
                raise FileNotFoundError("Model file '{}' or '{}' not found.".format(model_file_pt, model_file_pth))
        else:
            model_file = os.path.join(model_path, model + ".pkl")
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                return model
            else:
                raise FileNotFoundError("Model file '{}' not found.".format(model_file))

    def save_model(self, model_path):
        # Save model to specified location
        model_file = osp.join(
        model_path,
        '{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(self.cfg['run_config']['model'], self.cfg['run_config']['dataset'], self.cfg['run_config']['learning_rate'], self.cfg['run_config']['weight_decay'],
                                                      self.cfg['run_config']['batch_size'], self.cfg['run_config']['epochs'], self.cfg['run_config']['optimizer'], self.cfg['run_config']['lr_scheduler']))
        torch.save(self.model, model_file)

    def get_model(self, model_name, cv_n_split, random_state, test_train_split_part, model_type):
        if model_name == 'lr':
            X_train = self.dataset[0]['x'][self.dataset[0]['train_mask']]
            y_train = self.dataset[0]['y'][self.dataset[0]['train_mask']]
            X_test = self.dataset[0]['x'][self.dataset[0]['test_mask'] | self.dataset[0]['val_mask']]
            y_test = self.dataset[0]['y'][self.dataset[0]['test_mask'] | self.dataset[0]['val_mask']]
            if self.retrain:
                model = self.load_model(self.cfg['run_config']['model_path'], self.cfg['run_config']['model'])
            else:
                cv_train = ShuffleSplit(n_splits=cv_n_split, test_size=test_train_split_part, random_state=random_state)
                model = logistic_regression(self.cfg['lr_config']['param_grid_c'], cv_train, cv_n_split, X_train, y_train, X_test, y_test)
        if model_name == 'rf':
            X_train = self.dataset[0]['x'][self.dataset[0]['train_mask']]
            y_train = self.dataset[0]['y'][self.dataset[0]['train_mask']]
            X_test = self.dataset[0]['x'][self.dataset[0]['test_mask'] | self.dataset[0]['val_mask']]
            y_test = self.dataset[0]['y'][self.dataset[0]['test_mask'] | self.dataset[0]['val_mask']]
            if self.retrain:
                model = self.load_model(self.cfg['run_config']['model_path'], self.cfg['run_config']['model'])
            else:
                cv_train = ShuffleSplit(n_splits=cv_n_split, test_size=test_train_split_part, random_state=random_state)
                model = random_forest(self.cfg['rf_config']['param_grid_n_estimators'], self.cfg['rf_config']['param_grid_max_depth'], self.cfg['rf_config']['param_grid_bootstrap'], cv_train, cv_n_split, X_train, y_train, X_test, y_test)
        if model_name == 'xgb':
            X_train = self.dataset[0]['x'][self.dataset[0]['train_mask']]
            y_train = self.dataset[0]['y'][self.dataset[0]['train_mask']]
            X_test = self.dataset[0]['x'][self.dataset[0]['test_mask'] | self.dataset[0]['val_mask']]
            y_test = self.dataset[0]['y'][self.dataset[0]['test_mask'] | self.dataset[0]['val_mask']]
            if self.retrain:
                model = self.load_model(self.cfg['run_config']['model_path'], self.cfg['run_config']['model'])
            else:
                cv_train = ShuffleSplit(n_splits=cv_n_split, test_size=test_train_split_part, random_state=random_state)
                model = random_forest(self.cfg['xgb_config']['param_grid_n_estimators'], self.cfg['xgb_config']['param_grid_max_depth'], cv_train, cv_n_split, X_train, y_train, X_test, y_test)
        if model_name == 'bilstm':
            if self.retrain:
                model = self.load_model(self.cfg['run_config']['model_path'], self.cfg['run_config']['model'])
            else:
                model = BiLSTM(self.input_dim, self.cfg['bilstm_config']['hidden_dim'], self.output_dim, self.cfg['bilstm_config']['num_layers'], self.cfg['bilstm_config']['initializer'], self.cfg['bilstm_config']['dropout'], self.cfg['bilstm_config']['bn'], self.cfg['bilstm_config']['activation'], self.task, self.training)
        if model_name == 'resnet':
            if self.retrain:
                model = self.load_model(self.cfg['run_config']['model_path'], self.cfg['run_config']['model'])
            else:
                model = ResNet(self.input_dim, self.cfg['resnet_config']['hidden_dim'], self.output_dim, self.cfg['resnet_config']['num_layers'], self.cfg['resnet_config']['initializer'], self.cfg['resnet_config']['dropout'], self.cfg['resnet_config']['bn'], self.cfg['resnet_config']['activation'], self.task, self.training, self.cfg['run_config']['num_classes'])
        if model_name == 'mlp':
            if self.retrain:
                model = self.load_model(self.cfg['run_config']['model_path'], self.cfg['run_config']['model'])
            else:
                model = MLP(self.input_dim, self.cfg['mlp_config']['hidden_dim'], self.output_dim, self.cfg['mlp_config']['num_layers'], self.cfg['mlp_config']['initializer'], self.cfg['mlp_config']['dropout'], self.cfg['mlp_config']['bn'], self.cfg['mlp_config']['activation'], self.task, self.training)
        if model_name == 'gcn':
            if self.retrain:
                model = self.load_model(self.cfg['run_config']['model_path'], self.cfg['run_config']['model'])
            else:
                model = GCN(self.input_dim, self.cfg['gcn_config']['hidden_dim'], self.output_dim, self.cfg['gcn_config']['num_layers'], self.cfg['gcn_config']['initializer'], self.cfg['gcn_config']['dropout'], self.cfg['gcn_config']['bn'], self.cfg['gcn_config']['activation'], self.task, self.training, self.cfg['gcn_config']['gcn_layers'])
        if model_name == 'gat':
            if self.retrain:
                model = self.load_model(self.cfg['run_config']['model_path'], self.cfg['run_config']['model'])
            else:
                model = GAT(self.input_dim, self.cfg['gat_config']['hidden_dim'], self.output_dim, self.cfg['gat_config']['num_layers'], self.cfg['gat_config']['initializer'], self.cfg['gat_config']['dropout'], self.cfg['gat_config']['bn'], self.cfg['gat_config']['activation'], self.task, self.training, self.cfg['gat_config']['num_heads'], self.cfg['gat_config']['alpha'], self.cfg['gat_config']['gat_layers'])
        if model_name == 'sage':
            if self.retrain:
                model = self.load_model(self.cfg['run_config']['model_path'], self.cfg['run_config']['model'])
            else:
                model = SAGE(self.input_dim, self.cfg['sage_config']['hidden_dim'], self.output_dim, self.cfg['sage_config']['num_layers'], self.cfg['sage_config']['initializer'], self.cfg['sage_config']['dropout'], self.cfg['sage_config']['bn'], self.cfg['sage_config']['activation'], self.task, self.training, self.cfg['sage_config']['gcn_layers'])
        if model_name == 'graph_classification':
            if self.retrain:
                model = self.load_model(self.cfg['run_config']['model_path'], self.cfg['run_config']['model'])
            else:
                config_name = model_type + '_config'
                model = GraphClassifier(self.input_dim, self.cfg[config_name]['hidden_dim'], self.output_dim, self.cfg[config_name]['num_layers'], self.cfg[config_name]['initializer'], self.cfg[config_name]['dropout'], self.cfg[config_name]['bn'], self.cfg[config_name]['activation'], 'binary_classification', self.training, self.cfg[config_name]['gcn_layers'], model_type, self.cfg['run_config']['pooling'])
        return model  # Read model name from args
    
    def optimizer(self, model, weight_decay):
        if self.cfg['run_config']['model'] == 'resnet':
            model = model.resnet
        optimizer_name = self.cfg['run_config']['optimizer']
        if optimizer_name == 'adam':
            if weight_decay:
                optimizer = optim.Adam(model.parameters(), lr=self.cfg['run_config']['learning_rate'], weight_decay=weight_decay)
            else:
                optimizer = optim.Adam(model.parameters(), lr=self.cfg['run_config']['learning_rate'])
        elif optimizer_name == 'sgd':
            if weight_decay:
                optimizer = optim.SGD(model.parameters(), lr=self.cfg['run_config']['learning_rate'], momentum=self.cfg['run_config']['momentum'], weight_decay=weight_decay)
            else:
                optimizer = optim.SGD(model.parameters(), lr=self.cfg['run_config']['learning_rate'], momentum=self.cfg['run_config']['momentum'])
        elif optimizer_name == 'rmsprop':
            if weight_decay:
                optimizer = optim.RMSprop(model.parameters(), lr=self.cfg['run_config']['learning_rate'], weight_decay=weight_decay)
            else:
                optimizer = optim.RMSprop(model.parameters(), lr=self.cfg['run_config']['learning_rate'])
        else:
            raise ValueError(f'Unsupported optimizer: {optimizer_name}')
        
        # Other techniques like early stop, weight decay, etc., can be added here
        return optimizer
    
    def scheduler(self):
        if self.cfg['run_config']['lr_scheduler'] == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.cfg['run_config']['lr_decay_steps'], gamma=self.cfg['run_config']['lr_decay_rate'])
        elif self.cfg['run_config']['lr_scheduler'] == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.cfg['run_config']['lr_decay_rate'], patience=self.cfg['run_config']['lr_patience'], threshold=self.cfg['run_config']['lr_threshold'], threshold_mode='rel', cooldown=self.cfg['run_config']['lr_cooldown'], min_lr=self.cfg['run_config']['lr_decay_min_lr'])
        else:
            raise ValueError(f'Unsupported lr_scheduler: {self.cfg["run_config"]["lr_scheduler"]}')
        return scheduler
    
    def get_index(self, dataset):
        train_mask = dataset.data_list[0].train_mask
        val_mask = dataset.data_list[0].val_mask
        test_mask = dataset.data_list[0].test_mask
        graph_run_index = {'train': [0 for i in range(len(train_mask)*2)], 'val': [0 for i in range(len(train_mask)*2)], 'test': [0 for i in range(len(train_mask)*2)]}
        for i in range(len(train_mask)):
            if train_mask[i]:
                graph_run_index['train'][i] = 1
                graph_run_index['train'][i+len(train_mask)] = 1
            else:
                graph_run_index['train'][i] = 0
                graph_run_index['train'][i+len(train_mask)] = 0
            if val_mask[i]:
                graph_run_index['val'][i] = 1
                graph_run_index['val'][i+len(train_mask)] = 1
            else:
                graph_run_index['val'][i] = 0
                graph_run_index['val'][i+len(train_mask)] = 0
            if test_mask[i]:
                graph_run_index['test'][i] = 1
                graph_run_index['test'][i+len(train_mask)] = 1
            else:
                graph_run_index['test'][i] = 0
                graph_run_index['test'][i+len(train_mask)] = 0
        return graph_run_index
    
    def train(self):
        # Train the model
        print('starting one epoch training')
        self.clear_metrics()
        if self.cfg['run_config']['model'] == 'resnet':
            self.model.resnet.train()
        else:
            self.model.train()
        if self.cfg['run_config']['task'] == 'graph_classification' or self.cfg['run_config']['task'] == 'multignn':
            graph_run_index = self.get_index(self.dataset)
        if self.cfg['run_config']['dataloader']:
            # dataloader
            train_bar = tqdm(self.dataloader, ncols=70)
            batch_processing_times = []
            for count, batch in enumerate(train_bar):
                start_time = time.time()
                self.optimizer.zero_grad()
                if self.cfg['run_config']['graph']:
                    output = self.model(batch.x, batch.edge_index)
                else:
                    if self.cfg['run_config']['model'] == 'resnet':
                        output = self.model.forward(batch.x)
                    else:
                        output = self.model(batch.x)
                loss = self.model.loss(output, batch, 'train')
                loss.backward()
                self.optimizer.step()
                self.loss += loss.item()
                if self.cfg['run_config']['task'] == 'binary_classification' or self.cfg['run_config']['task'] == 'multi_classification' or self.cfg['run_config']['task'] == 'graph_classification' or self.cfg['run_config']['task'] == 'multignn':
                    accuracy, precision, recall, f1, auroc = self.metrics.calculate_metrics(batch.y[batch.train_mask], output[batch.train_mask], self.cfg['run_config']['model'], self.cfg['run_config']['task'])
                    self.accuracy += accuracy
                    self.precision += precision
                    self.recall += recall
                    self.f1 += f1
                    self.auroc += auroc
                elif self.cfg['run_config']['task'] == 'regression':
                    mse, mae, r2 = self.metrics.calculate_metrics(batch.y[batch.train_mask], output[batch.train_mask], self.cfg['run_config']['model'], self.cfg['run_config']['task'])
                    self.mse += mse
                    self.mae += mae
                    self.r2 += r2
                else:
                    raise ValueError(f'Unsupported task: {self.cfg["run_config"]["task"]}')
                batch_processing_times.append(time.time() - start_time)
            self.loss /= len(self.dataloader) # average batch loss
            if self.cfg['run_config']['task'] == 'binary_classification' or self.cfg['run_config']['task'] == 'multi_classification' or self.cfg['run_config']['task'] == 'graph_classification' or self.cfg['run_config']['task'] == 'multignn':
                self.accuracy /= len(self.dataloader)
                self.precision /= len(self.dataloader)
                self.recall /= len(self.dataloader)
                self.f1 /= len(self.dataloader)
                self.auroc /= len(self.dataloader)
            elif self.cfg['run_config']['task'] == 'regression':
                self.mse /= len(self.dataloader)
                self.mae /= len(self.dataloader)
                self.r2 /= len(self.dataloader)
            else:
                raise ValueError(f'Unsupported task: {self.cfg["run_config"]["task"]}')
            self.scheduler.step(loss)
            print('ending one epoch training')
            return np.mean(batch_processing_times)
        else:
            # graph classification GNN no dataloader
            if self.cfg['run_config']['task'] == 'graph_classification':
                graph_processing_times = []
                y_pred = []
                y_true = []
                for i in range(self.dataset.data_list[0].train_mask.shape[0], self.dataset.data_list[0].train_mask.shape[0]*2):
                    if graph_run_index['train'][i] == 1:
                        start_time = time.time()
                        self.optimizer.zero_grad()
                        output = self.model(self.dataset.data_list[i].x, self.dataset.data_list[i].edge_index)
                        loss = self.model.loss(output, self.dataset.data_list[i], 'train')
                        loss.backward()
                        self.optimizer.step()
                        self.loss = loss.item()
                        y_true = torch.tensor(y_true, dtype=torch.float)
                        y_true = self.dataset.data_list[i].y
                        y_true = torch.mean(y_true)
                        y_true = torch.full_like(y_pred, y_true)
                        y_true = y_true.squeeze(1)
                        y_pred = deepcopy(output)
                        if self.cfg['run_config']['task'] == 'binary_classification' or self.cfg['run_config']['task'] == 'multi_classification' or self.cfg['run_config']['task'] == 'graph_classification' or self.cfg['run_config']['task'] == 'multignn':
                            accuracy, precision, recall, f1, auroc = self.metrics.calculate_metrics(self.dataset.data_list[i].y, output, self.cfg['run_config']['model'], self.cfg['run_config']['task'])
                            self.accuracy += accuracy
                            self.precision += precision
                            self.recall += recall
                            self.f1 += f1
                            self.auroc += auroc
                        elif self.cfg['run_config']['task'] == 'regression':
                            mse, mae, r2 = self.metrics.calculate_metrics(self.dataset.data_list[i].y, output, self.cfg['run_config']['model'], self.cfg['run_config']['task'])
                            self.mse += mse
                            self.mae += mae
                            self.r2 += r2
                        else:
                            raise ValueError(f'Unsupported task: {self.cfg["run_config"]["task"]}')
                        graph_processing_times.append(time.time() - start_time)
                self.loss /= len(self.dataset.data_list[0].train_mask)
                if self.cfg['run_config']['task'] == 'binary_classification' or self.cfg['run_config']['task'] == 'multi_classification' or self.cfg['run_config']['task'] == 'graph_classification' or self.cfg['run_config']['task'] == 'multignn':
                    self.accuracy /= len(self.dataset.data_list[0].train_mask)
                    self.precision /= len(self.dataset.data_list[0].train_mask)
                    self.recall /= len(self.dataset.data_list[0].train_mask)
                    self.f1 /= len(self.dataset.data_list[0].train_mask)
                    self.auroc /= len(self.dataset.data_list[0].train_mask)
                elif self.cfg['run_config']['task'] == 'regression':
                    self.mse /= len(self.dataset.data_list[0].train_mask)
                    self.mae /= len(self.dataset.data_list[0].train_mask)
                    self.r2 /= len(self.dataset.data_list[0].train_mask)
                else:
                    raise ValueError(f'Unsupported task: {self.cfg["run_config"]["task"]}')
                self.scheduler.step(loss)
                print('ending one epoch training')
                return np.mean(graph_processing_times)
            else:
                # baseline GNN no dataloader
                start_time = time.time()
                self.optimizer.zero_grad()
                if self.cfg['run_config']['graph']:
                    output = self.model(self.dataset.data.x, self.dataset.data.edge_index)
                else:
                    if self.cfg['run_config']['model'] == 'resnet':
                        output = self.model.forward(self.dataset.data.x)
                    else:
                        output = self.model(self.dataset.data.x)
                loss = self.model.loss(output, self.dataset.data, 'train')
                loss.backward()
                self.optimizer.step()
                self.loss = loss.item()
                if self.cfg['run_config']['task'] == 'binary_classification' or self.cfg['run_config']['task'] == 'multi_classification' or self.cfg['run_config']['task'] == 'graph_classification' or self.cfg['run_config']['task'] == 'multignn':
                    accuracy, precision, recall, f1, auroc = self.metrics.calculate_metrics(self.dataset.data.y[self.dataset.data.train_mask], output[self.dataset.data.train_mask], self.cfg['run_config']['model'], self.cfg['run_config']['task'])
                    self.accuracy = accuracy
                    self.precision = precision
                    self.recall = recall
                    self.f1 = f1
                    self.auroc = auroc
                elif self.cfg['run_config']['task'] == 'regression':
                    mse, mae, r2 = self.metrics.calculate_metrics(self.dataset.data.y[self.dataset.data.train_mask], output[self.dataset.data.train_mask], self.cfg['run_config']['model'], self.cfg['run_config']['task'])
                    self.mse = mse
                    self.mae = mae
                    self.r2 = r2
                else:
                    raise ValueError(f'Unsupported task: {self.cfg["run_config"]["task"]}')
                processing_times = time.time() - start_time
                self.scheduler.step(loss)
                print('ending one epoch training')
                return processing_times
                
    
    def eval(self):
        # Evaluate the model
        print('starting one epoch evaluation')
        self.clear_metrics()
        if self.cfg['run_config']['model'] == 'resnet':
            self.model.resnet.eval()
        else:
            self.model.eval()
        self.training = False
        if self.cfg['run_config']['dataloader']:
            eval_bar = tqdm(self.dataloader, ncols=70)
            batch_processing_times = []
            with torch.no_grad():
                for count, batch in enumerate(eval_bar):
                    start_time = time.time()
                    if self.cfg['run_config']['graph']:
                        output = self.model(batch.x, batch.edge_index)
                    else:
                        if self.cfg['run_config']['model'] == 'resnet':
                            output = self.model.forward(batch.x)
                        else:
                            output = self.model(batch.x)
                    loss = self.model.loss(output, batch, 'val')
                    self.loss += loss.item()
                    batch_processing_times.append(time.time() - start_time)
                    if self.cfg['run_config']['task'] == 'binary_classification' or self.cfg['run_config']['task'] == 'multi_classification' or self.cfg['run_config']['task'] == 'graph_classification' or self.cfg['run_config']['task'] == 'multignn':
                        accuracy, precision, recall, f1, auroc = self.metrics.calculate_metrics(batch.y[batch.val_mask], output[batch.val_mask], self.cfg['run_config']['model'], self.cfg['run_config']['task'])
                        self.accuracy += accuracy
                        self.precision += precision
                        self.recall += recall
                        self.f1 += f1
                        self.auroc += auroc
                    elif self.cfg['run_config']['task'] == 'regression':
                        mse, mae, r2 = self.metrics.calculate_metrics(batch.y[batch.val_mask], output[batch.val_mask], self.cfg['run_config']['model'], self.cfg['run_config']['task'])
                        self.mse += mse
                        self.mae += mae
                        self.r2 += r2
                    else:
                        raise ValueError(f'Unsupported task: {self.cfg["run_config"]["task"]}')
            self.loss /= len(self.dataloader)
            if self.cfg['run_config']['task'] == 'binary_classification' or self.cfg['run_config']['task'] == 'multi_classification' or self.cfg['run_config']['task'] == 'graph_classification' or self.cfg['run_config']['task'] == 'multignn':
                self.accuracy /= len(self.dataloader)
                self.precision /= len(self.dataloader)
                self.recall /= len(self.dataloader)
                self.f1 /= len(self.dataloader)
                self.auroc /= len(self.dataloader)
            elif self.cfg['run_config']['task'] == 'regression':
                self.mse /= len(self.dataloader)
                self.mae /= len(self.dataloader)
                self.r2 /= len(self.dataloader)
            else:
                raise ValueError(f'Unsupported task: {self.cfg["run_config"]["task"]}')
            print('ending one epoch evaluation')
            return np.mean(batch_processing_times)
        else:
            start_time = time.time()
            with torch.no_grad():
                if self.cfg['run_config']['graph']:
                    output = self.model(self.dataset.data.x, self.dataset.data.edge_index)
                else:
                    if self.cfg['run_config']['model'] == 'resnet':
                        output = self.model.forward(self.dataset.data.x)
                    else:
                        output = self.model(self.dataset.data.x)
                loss = self.model.loss(output, self.dataset.data, 'val')
                self.loss = loss.item()
                if self.cfg['run_config']['task'] == 'binary_classification' or self.cfg['run_config']['task'] == 'multi_classification' or self.cfg['run_config']['task'] == 'graph_classification' or self.cfg['run_config']['task'] == 'multignn':
                    accuracy, precision, recall, f1, auroc = self.metrics.calculate_metrics(self.dataset.data.y[self.dataset.data.val_mask], output[self.dataset.data.val_mask], self.cfg['run_config']['model'], self.cfg['run_config']['task'])
                    self.accuracy = accuracy
                    self.precision = precision
                    self.recall = recall
                    self.f1 = f1
                    self.auroc = auroc
                elif self.cfg['run_config']['task'] == 'regression':
                    mse, mae, r2 = self.metrics.calculate_metrics(self.dataset.data.y[self.dataset.data.val_mask], output[self.dataset.data.val_mask], self.cfg['run_config']['model'], self.cfg['run_config']['task'])
                    self.mse = mse
                    self.mae = mae
                    self.r2 = r2
                else:
                    raise ValueError(f'Unsupported task: {self.cfg["run_config"]["task"]}')
            processing_times = time.time() - start_time
            print('ending one epoch evaluation')
            return processing_times
    
    def test(self):
        # Test the model
        print('starting one epoch testing')
        self.clear_metrics()
        if self.cfg['run_config']['model'] == 'resnet':
            self.model.resnet.eval()
        else:
            self.model.eval()
        self.training = False
        if self.cfg['run_config']['dataloader']:
            test_bar = tqdm(self.dataloader, ncols=70)
            batch_processing_times = []
            with torch.no_grad():
                for count, batch in enumerate(test_bar):
                    start_time = time.time()
                    if self.cfg['run_config']['graph']:
                        output = self.model(batch.x, batch.edge_index)
                    else:
                        if self.cfg['run_config']['model'] == 'resnet':
                            output = self.model.forward(batch.x)
                        else:
                            output = self.model(batch.x)
                    loss = self.model.loss(output, batch, 'test')
                    self.loss += loss.item()
                    batch_processing_times.append(time.time() - start_time)
                    if self.cfg['run_config']['task'] == 'binary_classification' or self.cfg['run_config']['task'] == 'multi_classification' or self.cfg['run_config']['task'] == 'graph_classification' or self.cfg['run_config']['task'] == 'multignn':
                        accuracy, precision, recall, f1, auroc = self.metrics.calculate_metrics(batch.y[batch.test_mask], output[batch.test_mask], self.cfg['run_config']['model'], self.cfg['run_config']['task'])
                        self.accuracy += accuracy
                        self.precision += precision
                        self.recall += recall
                        self.f1 += f1
                        self.auroc += auroc
                    elif self.cfg['run_config']['task'] == 'regression':
                        mse, mae, r2 = self.metrics.calculate_metrics(batch.y[batch.test_mask], output[batch.test_mask], self.cfg['run_config']['model'], self.cfg['run_config']['task'])
                        self.mse += mse
                        self.mae += mae
                        self.r2 += r2
                    else:
                        raise ValueError(f'Unsupported task: {self.cfg["run_config"]["task"]}')
            self.loss /= len(self.dataloader)
            if self.cfg['run_config']['task'] == 'binary_classification' or self.cfg['run_config']['task'] == 'multi_classification' or self.cfg['run_config']['task'] == 'graph_classification' or self.cfg['run_config']['task'] == 'multignn':
                self.accuracy /= len(self.dataloader)
                self.precision /= len(self.dataloader)
                self.recall /= len(self.dataloader)
                self.f1 /= len(self.dataloader)
                self.auroc /= len(self.dataloader)
            elif self.cfg['run_config']['task'] == 'regression':
                self.mse /= len(self.dataloader)
                self.mae /= len(self.dataloader)
                self.r2 /= len(self.dataloader)
            else:
                raise ValueError(f'Unsupported task: {self.cfg["run_config"]["task"]}')
            print('ending one epoch testing')
            return np.mean(batch_processing_times)
        else:
            start_time = time.time()
            with torch.no_grad():
                if self.cfg['run_config']['graph']:
                    output = self.model(self.dataset.data.x, self.dataset.data.edge_index)
                else:   
                    if self.cfg['run_config']['model'] == 'resnet':
                        output = self.model.forward(self.dataset.data.x)
                    else:
                        output = self.model(self.dataset.data.x)
                loss = self.model.loss(output, self.dataset.data, 'test')
                self.loss = loss.item()
                if self.cfg['run_config']['task'] == 'binary_classification' or self.cfg['run_config']['task'] == 'multi_classification' or self.cfg['run_config']['task'] == 'graph_classification' or self.cfg['run_config']['task'] == 'multignn':
                    accuracy, precision, recall, f1, auroc = self.metrics.calculate_metrics(self.dataset.data.y[self.dataset.data.test_mask], output[self.dataset.data.test_mask], self.cfg['run_config']['model'], self.cfg['run_config']['task'])
                    self.accuracy = accuracy
                    self.precision = precision
                    self.recall = recall
                    self.f1 = f1
                    self.auroc = auroc
                elif self.cfg['run_config']['task'] == 'regression':
                    mse, mae, r2 = self.metrics.calculate_metrics(self.dataset.data.y[self.dataset.data.test_mask], output[self.dataset.data.test_mask], self.cfg['run_config']['model'], self.cfg['run_config']['task'])
                    self.mse = mse
                    self.mae = mae
                    self.r2 = r2
                else:
                    raise ValueError(f'Unsupported task: {self.cfg["run_config"]["task"]}')
            processing_times = time.time() - start_time
            print('ending one epoch testing')
            return processing_times

    # def get_self_variables_as_dict(self):
    #     self_variables_dict = {}
    #     for key, value in self.__dict__.items():
    #         if not key.startswith('__') and not callable(value):
    #             self_variables_dict[key] = value
    #     json_string = json.dumps(self_variables_dict, skipkeys=True)
    #     self_variables_dict_json = json.loads(json_string)
    #     return self_variables_dict_json
    def get_self_variables_as_dict(self):
        self_variables_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('__') and not callable(value):
                # Convert non-JSON serializable data to string format
                if not isinstance(value, (int, float, str, bool, list, tuple, dict)):
                    value = str(value)
                # Convert torch device objects to string representation
                if isinstance(value, torch.device):
                    value = str(value)
                self_variables_dict[key] = value
        return self_variables_dict