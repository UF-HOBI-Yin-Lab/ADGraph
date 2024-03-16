# Install required packages.
import os
import torch
import torch.nn as nn
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

# !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
# !pip install -q captum

# Helper function for visualization.
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
from src.utils.config import Config
import sys
sys.path.insert(0, '/home/weimin.meng/projects/AD_progression')
os.chdir('/home/weimin.meng/projects/AD_progression')


data_path = '/blue/yonghui.wu/weimin.meng/AD_Progression/data/data/graph_mci_ad/processed/graph_mci_ad_0.4_346.8_660.dataset'
df_path = '/blue/yonghui.wu/weimin.meng/AD_Progression/data/data/graph_mci_ad/raw/graph_mci_ad_0.4_346.8_660.tsv'
model_path = '/blue/yonghui.wu/weimin.meng/AD_Progression/checkpoints/sage_mci_ad_0.001_0.0_16_100_adam_ReduceLROnPlateau.pkl'
age_df_path = 'src/model/ad_age.csv'
explaination = 'src/model/mci_ad_explaination.pkl'


mci_ad_graph = torch.load(data_path)
mci_ad_df = pd.read_csv(df_path, sep='\t')
mci_ad_model = torch.load(model_path)
age_df = pd.read_csv(age_df_path)

from torch_geometric.explain import Explainer, CaptumExplainer
explainer = Explainer(
    model=mci_ad_model, 
    algorithm=CaptumExplainer('IntegratedGradients'),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type=None,
    model_config=dict(
        mode='binary_classification',
        task_level='node',
        return_type='probs',
))

from torch_geometric.explain import Explainer, CaptumExplainer
shapexplainer = Explainer(
    model=mci_ad_model, 
    algorithm=CaptumExplainer('ShapleyValueSampling'),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type=None,
    model_config=dict(
        mode='binary_classification',
        task_level='node',
        return_type='probs',
))

from captum.attr import ShapleyValues, ShapleyValueSampling
shap = ShapleyValues(mci_ad_model)
mask = shap.attribute(input_mask, target=0, additional_forward_args=(mci_ad_graph[0],), internal_batch_size=(mci_ad_graph[0].edge_index.shape[1],))

from captum.attr import ShapleyValues, ShapleyValueSampling
import torch

x = mci_ad_graph[0].x 
edge_index = mci_ad_graph[0].edge_index 

x_tensor = torch.tensor(x, dtype=torch.float)
edge_index_tensor = torch.tensor(edge_index, dtype=torch.long) 

def forward_func(x, edge_index):
    output = mci_ad_model(x, edge_index)
    return output

shap = ShapleyValues(forward_func)

shap_values = shap.attribute(inputs=(x_tensor, edge_index_tensor))

# save the explaination to file
import pickle
with open('mci_ad_explaination.pkl', 'wb') as f:
    pickle.dump(explaination, f)

# mci_ad_explaination load as pickle
import pickle
with open(explaination, 'rb') as f:
    mci_ad_explaination = pickle.load(f)

node_mask = mci_ad_explaination.node_mask
# calculate node mask by summing the rows
node_importance = node_mask.sum(0)

mci_ad_df_age_merge = pd.merge(mci_ad_df, age_df, left_on='Participant ID', right_on='participant.eid', how='left')

# drop participant.eid and change participant.p21022 to age
mci_ad_df_age_merge = mci_ad_df_age_merge.drop(columns=['participant.eid'])
mci_ad_df_age_merge = mci_ad_df_age_merge.rename(columns={'participant.p21022': 'age'})
mci_ad_df_age_merge.head()

# add mci_ad_df['Time Between MCI and AD'] to mci_ad_graph[0].x as a new feature
mci_ad_graph[0].x = torch.cat([mci_ad_graph[0].x, torch.tensor(mci_ad_df_age_merge['age']).unsqueeze(1)], dim=1)
mci_ad_graph[0].x.shape


node_importance = node_importance.view(-1, 1)

# Concatenate tensors along dimension 1
mci_ad_graph[0].x = torch.cat([mci_ad_graph[0].x, node_importance], dim=1)

# Check the shape after concatenation
print(mci_ad_graph[0].x.shape)

# delete '| Instance 0' from the column names
mci_ad_df.columns = mci_ad_df.columns.str.replace('| Instance 0', '')
# strip the column names
mci_ad_df.columns = mci_ad_df.columns.str.strip()

import shap
shap.summary_plot(node_mask, mci_ad_df.drop(columns=['Participant ID', 'Time Between MCI and AD', 'label']))

# change explaination.node_mask as a dataframe, the columns are like in mci_ad_df
node_mask_df = pd.DataFrame(mci_ad_explaination.node_mask, columns=mci_ad_df.drop(columns=['Participant ID', 'Time Between MCI and AD', 'label']).columns)

for col in mci_ad_df.columns:
    if 'Time' in col:
        print(col)


import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import math

G = to_networkx(mci_ad_graph[0], to_undirected=True)

ages = mci_ad_graph[0].x[:, -1] 

age_color = ages.numpy()
max_age = np.max(age_color)
min_age = np.min(age_color)
norm = plt.Normalize(min_age, max_age)
cmap = plt.cm.coolwarm

plt.figure(figsize=(12, 8))
ax = plt.gca() 
pos = nx.spring_layout(G, k=1/math.sqrt(134)) 
nx.draw(G, pos, node_color=cmap(norm(age_color)), with_labels=False, node_size=150, edge_color="black", linewidths=0.05)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(age_color)

# cbar_ax = plt.axes([0.93, 0.15, 0.02, 0.7]) 
# # plt.colorbar(sm, cax=cbar_ax)
# cbar = plt.colorbar(sm, cax=cbar_ax)
# cbar.set_label('Year')  
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.045)
# cbar.set_label('Year', labelpad=10, rotation=270)

cbar.ax.set_title('Year', pad=10)

plt.show()


import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import matplotlib.colors as mcolors

# Convert to a networkx graph
G = to_networkx(mci_ad_graph[0], to_undirected=True)

# Get age attributes
ages = mci_ad_graph[0].x[:, -1]  # Assuming -1 is the index position of the age feature

# Create a mapping from age to color
age_color = ages.numpy()
max_age = np.max(age_color)
min_age = np.min(age_color)

# Create a PowerNorm normalization object
norm = mcolors.PowerNorm(gamma=0.4, vmin=min_age, vmax=max_age)

# Create a colormap using the PowerNorm normalization object
cmap = plt.cm.coolwarm_r

# Plot the graph
plt.figure(figsize=(12, 8))
ax = plt.gca()  
pos = nx.spring_layout(G, k=0.01)  
nx.draw(G, pos, node_color=cmap(norm(age_color)), with_labels=False, node_size=150, edge_color="black", linewidths=0.05)

# Create a ScalarMappable object for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(age_color)

# Create a colorbar
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.045)
cbar.ax.set_title('Year', pad=10, fontsize=15)

# Draw a red rectangle on the colorbar to highlight a specific range
highlight_start = norm(1.35)
highlight_end = norm(2.5)
rectangle_color = '#ffd700'
rectangle = plt.Rectangle((0, highlight_start), width=1, height=highlight_end-highlight_start, transform=cbar.ax.transAxes, color=rectangle_color, alpha=0.5)
cbar.ax.add_patch(rectangle)

# Add text indicating the highlighted range
buffer_position_y = (highlight_start + highlight_end) / 2
cbar.ax.text(x=-2.3, y=buffer_position_y, s='buffer', transform=cbar.ax.transAxes, color='black', ha='left', va='center', fontsize=15)
plt.savefig('/blue/yonghui.wu/weimin.meng/AD_Progression/data/data/img/progression_colorbar.png', dpi=300)
plt.show()

import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import matplotlib.colors as mcolors


# Convert to a networkx graph
G = to_networkx(mci_ad_graph[0], to_undirected=True)

# Get age attributes
ages = mci_ad_graph[0].x[:, -1]  # Assuming -1 is the index position of the age feature

# Create a mapping from age to color
age_color = ages.numpy()
max_age = np.max(age_color)
min_age = np.min(age_color)

# Create a PowerNorm normalization object
norm = mcolors.PowerNorm(gamma=1, vmin=min_age, vmax=max_age)

# Create a colormap
cmap = plt.cm.coolwarm

# Plot the graph
plt.figure(figsize=(12, 8))
ax = plt.gca()  # Get the current axis
pos = nx.spring_layout(G, k=0.01)  # Use different layout algorithms to see which one works best
nx.draw(G, pos, node_color=cmap(norm(age_color)), with_labels=False, node_size=150, edge_color="black", linewidths=0.05)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(age_color)

# Create a colorbar
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.045)
cbar.ax.set_title('Age', pad=10, fontsize=15)

# Draw a red rectangle on the colorbar to highlight a specific range
highlight_start = norm(1.35)
highlight_end = norm(2.5)
rectangle_color = '#ffd700'
rectangle = plt.Rectangle((0, highlight_start), width=1, height=highlight_end-highlight_start, transform=cbar.ax.transAxes, color=rectangle_color, alpha=0.5)
cbar.ax.add_patch(rectangle)

# Save the plot
plt.savefig('/blue/yonghui.wu/weimin.meng/AD_Progression/data/data/img/age_colorbar.png', dpi=300)
plt.show()

import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

# Convert to a networkx graph
G = to_networkx(mci_ad_graph[0], to_undirected=True)

# Get age attributes
ages = mci_ad_graph[0].x[:, -1]  # Assuming -1 is the index position of the age feature

# Log-transform the data
log_ages = np.log1p(ages.numpy())  # np.log1p performs log(x+1) transformation for each element

# Create a normalization object based on the log-transformed data
log_norm = plt.Normalize(np.min(log_ages), np.max(log_ages))

# Create a colormap
cmap = plt.cm.coolwarm

# Plot the graph
plt.figure(figsize=(12, 8))
ax = plt.gca()  # Get the current axis
pos = nx.spring_layout(G)  # Use different layout algorithms to see which one works best
nx.draw(G, pos, node_color=cmap(log_norm(log_ages)), with_labels=False, node_size=150, edge_color="black", linewidths=0.05)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=log_norm)
sm.set_array(log_ages)

# Create a colorbar
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.045)
cbar.ax.set_title('Age', pad=10, fontsize=15)

# Draw a red rectangle on the colorbar to highlight a specific range
log_highlight_start = np.log1p(1.35)  # Log-transform 1.35 years
log_highlight_end = np.log1p(2.5)      # Log-transform 2.5 years

# Calculate the relative positions of the log-transformed data
highlight_start = (log_highlight_start - np.min(log_ages)) / (np.max(log_ages) - np.min(log_ages))
highlight_end = (log_highlight_end - np.min(log_ages)) / (np.max(log_ages) - np.min(log_ages))

# Add a yellow rectangle on the colorbar to highlight the specified range
rectangle = plt.Rectangle((0, highlight_start), width=1, height=highlight_end-highlight_start, transform=cbar.ax.transAxes, color='yellow', alpha=0.5)
cbar.ax.add_patch(rectangle)

# Add a label for the highlighted range
buffer_position_y = (highlight_start + highlight_end) / 2
cbar.ax.text(x=-2.3, y=buffer_position_y, s='buffer', transform=cbar.ax.transAxes, color='black', ha='left', va='center', fontsize=15)

plt.show()
