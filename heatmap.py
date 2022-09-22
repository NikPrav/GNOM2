import seaborn as sns
import torch

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

plt.rcParams.update({'font.size': 20})


def plot_heatmap(data, vmin=-1., vmax=1., save_name='heatmap.pdf'):
    """ Plots a heatmap.

    :param data: DataFrame or ndarray.
    :param vmin:
    :param vmax:
    :param save_name:
    """

    fig, ax = plt.subplots()
    plt.figure(figsize=(11.7, 8.27))
    ax = sns.heatmap(data, vmin=vmin, vmax=vmax, annot=False, fmt="f",
                     linewidths=.5, cbar=True, cmap='viridis_r')
    plt.savefig(save_name, bbox_inches='tight')


x_gcn = torch.load('X_gcn.pt')
x_glove = torch.load('X_glove.pt')

words = {
    'earthquake': 2,
    'nepal':      6,
    'italy':      8,
    'relief':     21,
    'amatrice':   37,
    'government': 239,
    # 'terremoto':    51,
    'kathmandu':  53,
    'need':       61,
    # 'nepalquake':   119,
    'water':      141,
    # 'everest':      176,
    'iaf':        626,
    'damage':     106,
    'tent':       808,
    'medicine':   873,
    'available':  1050,
    # 'wifi':         2385,
    'building':   276,
    # 'recharge':     6415,
    # 'parbat':       16076,
    'donate':       83,
    'victims':       12,
    'help':       20,
    'italian':       43,
    # 'nepali':       390,
    'support':       102,
}

glove_tensor = []
gcn_tensor = []

for key in sorted(words):
    glove_tensor += [x_glove[words[key]].tolist()]
    gcn_tensor += [x_gcn[words[key]].tolist()]

gcn_sim = cosine_similarity(gcn_tensor)
gcn_normed = (gcn_sim - gcn_sim.mean()) / gcn_sim.std()

glove_sim = cosine_similarity(glove_tensor)
glove_normed = (glove_sim - glove_sim.mean()) / glove_sim.std()

index = sorted(words)
columns = sorted(words)

# gcn_sigm = torch.sigmoid(torch.from_numpy(gcn_normed)).numpy()
gcn_df = pd.DataFrame(gcn_normed, index=index, columns=columns)
gcn_df.to_csv('gcn_sim_df.csv')
plot_heatmap(data=gcn_df, save_name='gcn_heatmap.pdf')

# glove_sigm = torch.sigmoid(torch.from_numpy(glove_normed)).numpy()
glove_df = pd.DataFrame(glove_normed, index=index, columns=columns)
glove_df.to_csv('glove_sim_df.csv')
plot_heatmap(data=glove_df, save_name='glove_heatmap.pdf')

torch.save(torch.Tensor(gcn_sim), 'gcn_cosine_sim.pt')
torch.save(torch.Tensor(glove_sim), 'glove_cosine_sim.pt')
