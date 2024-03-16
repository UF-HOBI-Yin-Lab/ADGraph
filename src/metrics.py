import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score

def regression_scaling(y):
    '''
    Scale the target variable for regression task
    '''
    if (y < 0).any() or (y > 1).any():
        scaler = MinMaxScaler()
        y = scaler.fit_transform(y.reshape(-1, 1))
        # reshape
        y = torch.from_numpy(y.reshape(-1)).float()
    return y

class BinaryClassificationMetrics():
    def __init__(self):
        pass

    def calculate_metrics(self, y_true, y_pred, model, task):
        y_pred = torch.tensor(y_pred, dtype=torch.float)
        y_true = torch.tensor(y_true, dtype=torch.float)
        if y_pred.size(1) == 1:
            y_true = torch.mean(y_true)
            y_true = torch.full_like(y_pred, y_true)
            y_true = y_true.squeeze(1)

        if model == 'resnet':
            y_pred_binary = torch.argmax(y_pred, dim=1)
        else:
            # Convert probabilities to binary predictions
            y_pred_binary = torch.round(y_pred).cpu().detach().numpy()
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred_binary)
        
        # Calculate precision
        precision = precision_score(y_true, y_pred_binary)
        
        # Calculate recall
        recall = recall_score(y_true, y_pred_binary)
        
        # Calculate F1 score
        f1 = f1_score(y_true, y_pred_binary)
        
        # Calculate AUROC
        # y_pred_prob = F.softmax(torch.tensor(y_pred, dtype=torch.float), dim=1)[:, 0].cpu().detach().numpy()
        if model == 'resnet':
            auroc = roc_auc_score(y_true, y_pred_binary)
        elif task == 'graph_classification':
            auroc = -1
        else:
            auroc = roc_auc_score(y_true, y_pred)
        # auroc = roc_auc_score(y_true, y_pred_binary)
        
        return accuracy, precision, recall, f1, auroc
    

class RegressionMetrics():
    def __init__(self):
        pass

    def calculate_metrics(self, y_true, y_pred, model):
        y_true = regression_scaling(y_true)

        y_pred = torch.tensor(y_pred, dtype=torch.float)
        y_true = torch.tensor(y_true, dtype=torch.float)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_true, y_pred)
        
        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate R-squared (R2) score
        r2 = r2_score(y_true, y_pred)
        
        # Optionally, you can calculate other regression metrics
        
        return mse, mae, r2