from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models


class logistic_regression():
    def __init__(self, param_grid_c, cv_train, cv_n_split, X_train, y_train, X_test, y_test):
        self.param_grid_c = param_grid_c # [.2, .3, .4]
        self.cv_train = cv_train  # cv_train = ShuffleSplit(n_splits=cv_n_split[i], test_size=test_train_split_part[k], random_state=random_state[m])
        self.cv_n_split = cv_n_split
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.lr = LogisticRegression()
    
    def train(self):
        # Logistic Regression
        lr_CV = GridSearchCV(estimator=self.lr, param_grid={'C' : self.param_grid_c}, cv=self.cv_train, verbose=False)
        lr_CV.fit(self.X_train, self.y_train)
        print(lr_CV.best_params_)

    def test(self):
        # Logistic Regression
        train_pred = self.lr.predict(self.X_train)
        test_pred = self.lr.predict(self.X_test)
        return train_pred, test_pred
    

class random_forest():
    def __init__(self, param_grid_n_estimators, param_grid_max_depth, param_grid_bootstrap, cv_train, cv_n_split, X_train, y_train, X_test, y_test):
        self.param_grid_n_estimators = param_grid_n_estimators
        self.param_grid_max_depth = param_grid_max_depth
        self.param_grid_bootstrap = param_grid_bootstrap
        self.cv_train = cv_train  # cv_train = ShuffleSplit(n_splits=cv_n_split[i], test_size=test_train_split_part[k], random_state=random_state[m])
        self.cv_n_split = cv_n_split
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.rf = RandomForestClassifier()
    
    def train(self):
        # Random Forest
        rf_CV = GridSearchCV(estimator=self.rf, param_grid={'n_estimators' : self.param_grid_n_estimators, 'max_depth' : self.param_grid_max_depth, 'bootstrap': self.param_grid_bootstrap}, cv=self.cv_train, verbose=False)
        rf_CV.fit(self.X_train, self.y_train)
        print(rf_CV.best_params_)

    def test(self):
        # Random Forest
        train_pred = self.rf.predict(self.X_train)
        test_pred = self.rf.predict(self.X_test)
        return train_pred, test_pred
    

class xgboost():
    def __init__(self, param_grid_n_estimators, param_grid_max_depth, param_learning_rate, cv_train, cv_n_split, X_train, y_train, X_test, y_test):
        self.param_grid_n_estimators = param_grid_n_estimators
        self.param_grid_max_depth = param_grid_max_depth
        self.param_learning_rate = param_learning_rate
        self.cv_train = cv_train  # cv_train = ShuffleSplit(n_splits=cv_n_split[i], test_size=test_train_split_part[k], random_state=random_state[m])
        self.cv_n_split = cv_n_split
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.xgb = XGBClassifier()
    
    def train(self):
        # XGBoost
        xgb_CV = GridSearchCV(estimator=self.xgb, param_grid={'n_estimators' : self.param_grid_n_estimators, 'max_depth' : self.param_grid_max_depth, 'learning_rate': self.param_learning_rate}, cv=self.cv_train, verbose=False)
        xgb_CV.fit(self.X_train, self.y_train)
        print(xgb_CV.best_params_)

    def test(self):
        # XGBoost
        train_pred = self.xgb.predict(self.X_train)
        test_pred = self.xgb.predict(self.X_test)
        return train_pred, test_pred
    

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, initializer, dropout, bn, activation, task, training):
        super(BiLSTM, self).__init__()
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
        
        # BiLSTM
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=self.dropout, batch_first=True, bidirectional=True)
        # Dense Layer
        self.fc = nn.Linear(self.hidden_dim*2, self.output_dim)
        
        # Initialize weights
        self.weights_init()

    def forward(self, x):
        # Hidden state
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).to(x.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).to(x.device)
        
        x = x.unsqueeze(1)

        # forward
        x, (h1, c1) = self.lstm(x, (h0, c0))  # _ means hidden state
        
        # last time step
        x = self.fc(x[:, -1, :])

        x = torch.sigmoid(x)
        return x
    
    def weights_init(self):
        if self.initializer == 'xavier':
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
        elif self.initializer == 'he':
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
        else:
            raise ValueError('Initializer not understood')

    def reset_parameters(self):
        # Reset all learnable parameters
        # self.weights_init()
        # use reset_parameters() method
        for name, module in self.named_children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def loss(self, outputs, targets, mode):
        pred = outputs
        label, mask = targets['y'], targets[f'{mode}_mask']
        label = label.float()
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
            loss = F.mse_loss(pred[mask], label[mask].unsqueeze(1))
            return loss
        else:
            raise ValueError('Task not understood')


class ResNet():
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, initializer, dropout, bn, activation, task, training, num_classes):
        super(ResNet, self).__init__()
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
        self.num_classes = num_classes

        # Load a pre-trained ResNet model
        self.resnet = models.resnet18(pretrained=True)
        # Modify the first layer to accept single-channel (grayscale) images
        # self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the fully connected layer to output num_classes values for multi-class classification
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        # self.resnet.weight.requires_grad = True
        # for param in self.resnet.parameters():
        #     param.requires_grad = False 

        # Initialize weights
        # self.weights_init()

    def forward(self, x):
        # Reshape input data to fit the ResNet model
        if len(x.shape) == 2:
            # x = x.unsqueeze(1).unsqueeze(-1)
            x_expanded = x.unsqueeze(1)
            x_3d = x_expanded.repeat(1, x.size(1), 1)
            x_3d_expanded = x_3d.unsqueeze(1)
            x = x_3d_expanded.repeat(1, 3, 1, 1)
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)
        else:
            raise ValueError('Input data shape not understood')
        x = self.resnet(x)
        return x

    def weights_init(self):
        # No custom weight initialization needed, as the ResNet model is pre-trained
        pass

    def reset_parameters(self):
        # No need to reset parameters for pre-trained ResNet model
        pass

    def loss(self, outputs, targets, mode):
        pred = outputs
        # pred = torch.sigmoid(pred[:, 0]).detach()
        # pred = (pred >= 0.5).float().view(-1, 1)
        label, mask = targets['y'], targets[f'{mode}_mask']
        # label = label.float()
        # change 1 to 2, change 0 to 1 in label
        # label = label + 1
        label_one_hot = torch.zeros(len(label), 2)
        label_one_hot.scatter_(1, label.unsqueeze(1), 1)
        if self.task == 'binary_classification':
            # Compute binary cross-entropy loss
            loss = nn.BCEWithLogitsLoss()(pred[mask], label_one_hot[mask])
            # loss = nn.CrossEntropyLoss(pred[mask], label[mask]) # .unsqueeze(1)
            return loss
        elif self.task == 'multi_classification':
            # Compute multi-class cross-entropy loss
            loss = nn.CrossEntropyLoss()(pred[mask], label_one_hot[mask])
            return loss
        elif self.task == 'regression':
            # Compute mean squared error loss
            # loss = F.mse_loss(pred[mask], label[mask])
            # return loss
            raise ValueError('Regression not supported for ResNet')
        else:
            raise ValueError('Task not understood')