import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from src.model.metrics import regression_scaling

class MLP(nn.Module):
    '''
    The basic neural network model, DNN model
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, initializer, dropout, bn, activation, task, training): # hidden_dim = 32, output_dim = 1
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.initializer = initializer
        self.dropout = dropout
        self.bn = bn
        self.activation = activation
        self.task = task
        self.training = training

        self.fcs = nn.ModuleList([nn.Linear(self.input_dim if i == 0 else self.hidden_dim, self.hidden_dim) for i in range(self.num_layers)])
        self.fcs.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.relu = nn.ReLU()  # ReLU activation function
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function
        
        # Initialize weights using He initialization
        self.weights_init()

    def forward(self, x):
        for fc in self.fcs[:-1]:
            x = self.relu(fc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fcs[-1](x)
        x = self.sigmoid(x)
        return x
    
    def weights_init(self):
        if self.initializer == 'xavier':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
        elif self.initializer == 'he':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.zero_()
        else:
            raise ValueError('Initializer not understood')

    def reset_parameters(self):
        # Reset all learnable parameters
        # self.weights_init()
        for fc in self.fcs:
            fc.reset_parameters()

    def loss(self, outputs, targets, mode):
        pred = outputs
        label, mask = targets['y'], targets[f'{mode}_mask']
        label = label.float()
        # if pred.dim() == 1:
        #     # return F.binary_cross_entropy_with_logits(pred[mask], label[mask].unsqueeze(1).float())
        #     return F.binary_cross_entropy(pred[mask], label[mask].unsqueeze(1))
        # elif pred.dim() == 2:
        #     return F.cross_entropy(pred[mask], label[mask].unsqueeze(1))
        if self.task == 'binary_classification':
            # Compute binary cross-entropy loss
            loss = F.binary_cross_entropy(pred[mask], label[mask].unsqueeze(1))
            return loss
        elif self.task == 'multi_classification':
            # Compute multi-class cross-entropy loss
            loss = F.cross_entropy(pred[mask], label[mask].unsqueeze(1))
            return loss
        elif self.task == 'regression':
            # Compute mean squared error loss
            y_true = regression_scaling(label[mask])
            loss = F.mse_loss(pred[mask], y_true.unsqueeze(1))
            return loss
        else:
            raise ValueError('Task not understood')


class MLP_Regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, initializer, dropout, bn, activation, task, training):
        super(MLP_Regressor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.initializer = initializer
        self.dropout = dropout
        self.bn = bn
        self.activation = activation
        self.task = task
        self.training = training

        self.fcs = nn.ModuleList([nn.Linear(self.input_dim if i == 0 else self.hidden_dim, self.hidden_dim) for i in range(self.num_layers)])
        self.fcs.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.relu = nn.ReLU()  # ReLU activation function
        
        # Initialize weights using specified initialization method
        self.weights_init()

    def forward(self, x):
        for fc in self.fcs[:-1]:
            x = self.relu(fc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return x
    
    def weights_init(self):
        if self.initializer == 'xavier':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
        elif self.initializer == 'he':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.zero_()
        else:
            raise ValueError('Initializer not understood')

    def reset_parameters(self):
        # Reset all learnable parameters
        for fc in self.fcs:
            fc.reset_parameters()

    def loss(self, outputs, targets, mode):
        pred = outputs
        label, mask = targets['y'], targets[f'{mode}_mask']
        label = label.float()
        
        # Compute mean squared error loss
        loss = F.mse_loss(pred[mask], label[mask].unsqueeze(1))
        return loss


class GCN(nn.Module):
    '''
    The basic GCN model
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, initializer, dropout, bn, activation, task, training, gcn_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.initializer = initializer
        self.dropout = dropout
        self.bn = bn
        self.activation = activation
        self.task = task
        self.training = training
        self.gcn_layers = gcn_layers

        self.convs = nn.ModuleList([GCNConv(self.input_dim if i == 0 else self.hidden_dim, self.hidden_dim) for i in range(self.gcn_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.hidden_dim) for _ in range(self.gcn_layers - 1)]) if self.bn else None
        # self.fc = nn.Linear(hidden_dim, output_dim)
        # self.sigmoid = nn.Sigmoid()  # Sigmoid activation function
        if self.task == 'binary_classification':
            self.mlp_classifier = MLP(self.hidden_dim, self.hidden_dim, self.output_dim, self.num_layers, self.initializer, self.dropout, self.bn, self.activation, self.task, self.training)
        elif self.task == 'multi_classification':
            self.mlp_classifier = MLP(self.hidden_dim, self.hidden_dim, self.output_dim, self.num_layers, self.initializer, self.dropout, self.bn, self.activation, self.task, self.training)
        elif self.task == 'regression':
            self.mlp_regressor = MLP_Regressor(self.hidden_dim, self.hidden_dim, self.output_dim, self.num_layers, self.initializer, self.dropout, self.bn, self.activation, self.task, self.training)
        else:
            raise ValueError('Task not understood')
        
        # Initialize weights using He initialization
        self.weights_init()

    def weights_init(self):
        if self.initializer == 'xavier':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
        elif self.initializer == 'he':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.zero_()
        else:
            raise ValueError('Initializer not understood')
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.mlp_classifier.reset_parameters()


    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for layer in range(self.gcn_layers-1):
            x = self.convs[layer](x, edge_index)
            if self.bns is not None:
                x = self.bns[layer](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.convs[-1](x, edge_index))
        if self.bns is not None:
            x = self.bns[-1](x)
        # x = self.fc(x)
        # x = self.sigmoid(x)
        # return x.squeeze()
        if self.task == 'binary_classification' or self.task == 'multi_classification':
            x = self.mlp_classifier(x)
        elif self.task == 'regression':
            x = self.mlp_regressor(x)
        else:
            raise ValueError('Task not understood')
        return x
    
    def loss(self, outputs, targets, mode):
        pred = outputs
        label, mask = targets['y'], targets[f'{mode}_mask']
        label = label.float()
        # if pred.dim() == 1:
        #     # return F.binary_cross_entropy_with_logits(pred[mask], label[mask].unsqueeze(1).float())
        #     return F.binary_cross_entropy(pred[mask], label[mask].unsqueeze(1))
        # elif pred.dim() == 2:
        #     return F.cross_entropy(pred[mask], label[mask].unsqueeze(1))
        if self.task == 'binary_classification':
            # Compute binary cross-entropy loss
            loss = F.binary_cross_entropy(pred[mask], label[mask].unsqueeze(1))
            return loss
        elif self.task == 'multi_classification':
            # Compute multi-class cross-entropy loss
            loss = F.cross_entropy(pred[mask], label[mask].unsqueeze(1))
            return loss
        elif self.task == 'regression':
            # Compute mean squared error loss
            y_true = regression_scaling(label[mask])
            loss = F.mse_loss(pred[mask], y_true.unsqueeze(1))
            return loss
        else:
            raise ValueError('Task not understood')


class GAT(nn.Module):
    '''
    The basic Graph Attention Network (GAT) model
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, initializer, dropout, bn, activation, task, training, num_heads, alpha, gat_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.initializer = initializer
        self.dropout = dropout
        self.bn = bn
        self.activation = activation
        self.task = task
        self.training = training
        self.num_heads = num_heads
        self.alpha = alpha
        self.gat_layers = gat_layers

        self.gats = nn.ModuleList([GATConv(self.input_dim if i == 0 else self.hidden_dim * self.num_heads, 
                                             self.hidden_dim, heads=self.num_heads, concat=True, dropout=self.dropout) 
                                     for i in range(self.gat_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.hidden_dim * self.num_heads) for _ in range(self.gat_layers - 1)]) if self.bn else None
        self.fc = nn.Linear(self.hidden_dim * self.num_heads, self.output_dim)
        self.softmax = nn.Softmax(dim=1)

        self.weights_init()

    def weights_init(self):
        if self.initializer == 'xavier':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
        elif self.initializer == 'he':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.zero_()
        else:
            raise ValueError('Initializer not understood')
    
    def reset_parameters(self):
        for gat in self.gats:
            gat.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for layer in range(self.gat_layers-1):
            x = self.gats[layer](x, edge_index)
            if self.bns is not None:
                x = self.bns[layer](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gats[-1](x, edge_index)
        if self.bns is not None:
            x = self.bns[-1](x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)  # Global pooling
        # x = self.fc(x)
        # x = self.softmax(x * self.alpha)  # Attention mechanism
        x = torch.sigmoid(x * self.alpha)
        return x
    
    def loss(self, outputs, targets, mode):
        pred = outputs
        label, mask = targets['y'], targets[f'{mode}_mask']
        label = label.float()
        # if pred.dim() == 1:
        #     return F.binary_cross_entropy(pred[mask], label[mask].unsqueeze(1))
        # elif pred.dim() == 2:
        #     return F.cross_entropy(pred[mask], label[mask].unsqueeze(1))
        if self.task == 'binary_classification':
            # Compute binary cross-entropy loss
            loss = F.binary_cross_entropy(pred[mask], label[mask].unsqueeze(1))
            return loss
        elif self.task == 'multi_classification':
            # Compute multi-class cross-entropy loss
            loss = F.cross_entropy(pred[mask], label[mask].unsqueeze(1))
            return loss
        elif self.task == 'regression':
            # Compute mean squared error loss
            y_true = regression_scaling(label[mask])
            loss = F.mse_loss(pred[mask], y_true.unsqueeze(1))
            return loss
        else:
            raise ValueError('Task not understood')


class SAGE(nn.Module):
    '''
    The basic GraphSAGE model
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, initializer, dropout, bn, activation, task, training, gcn_layers):
        super().__init__()
        # self.conv1 = SAGEConv(input_dim, hidden_dim, normalize=True)
        # self.conv1.aggr = 'mean' # 'add', 'mean', 'max'
        # self.dp = dropout
        # self.transition = nn.Sequential(
        #     nn.ReLU(),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.Dropout(p=self.dp)
        # )
        # self.conv2 = SAGEConv(hidden_dim, hidden_dim, normalize=True)
        # self.conv2.aggr = 'mean'
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.initializer = initializer
        self.dropout = dropout
        self.bn = bn
        self.activation = activation
        self.task = task
        self.training = training
        self.gcn_layers = gcn_layers

        self.convs = nn.ModuleList([SAGEConv(self.input_dim if i == 0 else self.hidden_dim, self.hidden_dim) 
                                     for i in range(self.gcn_layers)])
        for conv in self.convs:
            conv.aggr = 'mean'
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.hidden_dim) for _ in range(self.gcn_layers - 1)]) if self.bn else None
        # self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        # self.sigmoid = nn.Sigmoid()  # Sigmoid activation function
        # self.mlp_classifier = MLP(self.hidden_dim, self.hidden_dim, self.output_dim, self.num_layers, self.initializer, self.dropout, self.bn, self.activation, self.task, self.training)
        if self.task == 'binary_classification':
            self.mlp_classifier = MLP(self.hidden_dim, self.hidden_dim, self.output_dim, self.num_layers, self.initializer, self.dropout, self.bn, self.activation, self.task, self.training)
        elif self.task == 'multi_classification':
            self.mlp_classifier = MLP(self.hidden_dim, self.hidden_dim, self.output_dim, self.num_layers, self.initializer, self.dropout, self.bn, self.activation, self.task, self.training)
        elif self.task == 'regression':
            self.mlp_regressor = MLP_Regressor(self.hidden_dim, self.hidden_dim, self.output_dim, self.num_layers, self.initializer, self.dropout, self.bn, self.activation, self.task, self.training)
        else:
            raise ValueError('Task not understood')
        
        # Initialize weights using He initialization
        self.weights_init()

    def weights_init(self):
        if self.initializer == 'xavier':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
        elif self.initializer == 'he':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.zero_()
        else:
            raise ValueError('Initializer not understood')
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.mlp_classifier.reset_parameters()

    def forward(self, x, edge_index):
        # x = self.conv1(x, edge_index)
        # x = self.transition(x)
        # x = self.conv2(x, edge_index)
        # x = self.fc(x)
        # x = self.sigmoid(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        for layer in range(self.gcn_layers-1):
            x = self.convs[layer](x, edge_index)
            if self.bns is not None:
                x = self.bns[layer](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.convs[-1](x, edge_index))
        if self.bns is not None:
            x = self.bns[-1](x)
        # x = self.fc(x)
        # x = self.sigmoid(x)
        # x = self.mlp_classifier(x)
        if self.task == 'binary_classification' or self.task == 'multi_classification':
            x = self.mlp_classifier(x)
        elif self.task == 'regression':
            x = self.mlp_regressor(x)
        else:
            raise ValueError('Task not understood')
        return x
    
    def loss(self, outputs, targets, mode):
        pred = outputs
        label, mask = targets['y'], targets[f'{mode}_mask']
        label = label.float()
        # if pred.dim() == 1:
        #     return F.binary_cross_entropy(pred[mask], label[mask].unsqueeze(1))
        # elif pred.dim() == 2:
        #     return F.cross_entropy(pred[mask], label[mask].unsqueeze(1))
        if self.task == 'binary_classification':
            # Compute binary cross-entropy loss
            loss = F.binary_cross_entropy(pred[mask], label[mask].unsqueeze(1))
            return loss
        elif self.task == 'multi_classification':
            # Compute multi-class cross-entropy loss
            loss = F.cross_entropy(pred[mask], label[mask].unsqueeze(1))
            return loss
        elif self.task == 'regression':
            # Compute mean squared error loss
            y_true = regression_scaling(label[mask])
            loss = F.mse_loss(pred[mask], y_true.unsqueeze(1))
            return loss
        else:
            raise ValueError('Task not understood')