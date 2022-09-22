import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity

from Utils.utils import merge_dicts


# 'nepal' = 6
# 'italy' = 8
# 'earthquake' = 2
# G.nodes[6]
# Out[75]: {'node_txt': 'nepal', 's_co': 34627, 't_co': 108}
# G.nodes[8]
# Out[76]: {'node_txt': 'italy', 's_co': 18, 't_co': 27224}
# G.nodes[2]
# Out[77]: {'node_txt': 'earthquake', 's_co': 50755, 't_co': 43121}
# C_vocab['idx2str_map'][2]
# Out[69]: 'earthquake'
# G[2][6]
# Out[70]: {'s_pair': 16994, 't_pair': 32, 'weight': 0.09988641050238399}
# G[2][8]
# Out[71]: {'s_pair': 1, 't_pair': 7784, 'weight': 0.055336376431352315}
# G[6][8]
# Out[78]: {'s_pair': 0, 't_pair': 2, 'weight': 3.658581202209783e-05}
#
# from sklearn.metrics.pairwise import cosine_similarity
#
# cosine_similarity(nepal_glove.reshape(1,-1), italy_glove.reshape(1,-1))
# Out[98]: array([[0.09416067]], dtype=float32)
#
# cosine_similarity(nepal_GCN.detach().unsqueeze(0), italy_GCN.detach().unsqueeze(0))
# Out[96]: array([[0.9516211]], dtype=float32)


def get_sets(S_vocab, limit=5000, reverse=True):
    S_sorted = sorted(S_vocab['freqs'].items(), key=lambda x: x[1], reverse=reverse)[:limit]
    S_bag = [token for (token, freq) in S_sorted if token in S_vocab['str2idx_map']]
    S_set = set(S_bag)
    return S_set


def get_freq_disjoint_token_vecs(S_vocab, T_vocab, X, limit=1000, reverse=True):
    S_diff = get_sets(S_vocab, limit=limit, reverse=reverse)
    T_diff = get_sets(T_vocab, limit=limit, reverse=reverse)
    # S_diff = S_diff - T_diff
    # T_diff = T_diff - S_diff
    S_diff = list(S_diff)
    T_diff = list(T_diff)

    S_idx = OrderedDict()
    S_diff = list(S_diff)[:limit]
    for token in S_diff:
        S_idx[token] = S_vocab['str2idx_map'][token]

    T_idx = OrderedDict()
    T_diff = list(T_diff)[:limit]
    for token in T_diff:
        T_idx[token] = T_vocab['str2idx_map'][token]

    S_vecs = OrderedDict()
    for token, idx in S_idx.items():
        S_vecs[token] = X[idx].numpy()

    T_vecs = OrderedDict()
    for token, idx in T_idx.items():
        T_vecs[token] = X[idx].numpy()

    return S_vecs, T_vecs


def heatmap_dict(S_vecs_dict, T_vecs_dict=None):
    S_ours_vecs = np.array(list(S_vecs_dict.values()))
    if T_vecs_dict is not None:
        T_ours_vecs = np.array(list(T_vecs_dict.values()))
        ours_sims = cosine_similarity(S_ours_vecs, T_ours_vecs)
        ours_sims_df = pd.DataFrame(data=ours_sims, index=list(
            S_vecs_dict.keys()), columns=list(T_vecs_dict.keys()))
    else:
        ours_sims = cosine_similarity(S_ours_vecs)
        ours_sims_df = pd.DataFrame(data=ours_sims, index=list(
            S_vecs_dict.keys()), columns=list(S_vecs_dict.keys()))
    return ours_sims_df


from Plotter.plot_functions import plot_heatmap


# glove_least_vecs_dict_S, glove_least_vecs_dict_T = get_freq_disjoint_token_vecs(
#     S_vocab, T_vocab, X, limit=100, reverse=False)
# glove_nonfrequent_sims_df = heatmap_dict(glove_least_vecs_dict_S, glove_least_vecs_dict_T)
# plot_heatmap(glove_nonfrequent_sims_df, save_name='GloVe_nonfrequent_token_heatmap.pdf')
#
# gcn_least_vecs_dict_S, gcn_least_vecs_dict_T = get_freq_disjoint_token_vecs(
#     S_vocab, T_vocab, token_embs.detach(), limit=100, reverse=False)
# gcn_nonfrequent_sims_df = heatmap_dict(gcn_least_vecs_dict_S, gcn_least_vecs_dict_T)
# plot_heatmap(gcn_nonfrequent_sims_df, save_name='GCN_nonfrequent_token_heatmap.pdf')


def plot_vecs_color(S_vecs, T_vecs, lim=None, save_name='tsne_vecs.pdf',
                    title='TSNE visualization of top 100 tokens in 2D'):
    """ Plots TSNE representations of tokens and their embeddings.

    :param X:
    :param tokens:
    :param limit_view:
    """
    tsne = TSNE(n_components=2, random_state=0)

    merged_vecs = merge_dicts(S_vecs, T_vecs)
    X = np.stack(list(merged_vecs.values()))
    tokens = list(merged_vecs.keys())
    tokens = None

    X_2d = tsne.fit_transform(X)
    s_c = ['g'] * len(S_vecs)
    t_c = ['b'] * len(T_vecs)
    colors = s_c + t_c
    # print(len(colors), colors)

    plt.figure(figsize=(15, 15))
    if tokens is not None:
        for i, token in enumerate(tokens):
            plt.annotate(token, xy=(X_2d[i, 0], X_2d[i, 1]), zorder=1)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=60, alpha=.5)
    if lim is not None:
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
    plt.tight_layout()
    plt.title(title)
    # plt.xlabel('x-axis')
    # plt.ylabel('y-axis')
    plt.savefig(save_name)
    plt.show()


def plot(tokens2vec):
    """ Plots TSNE representations of tokens and their embeddings.

    :param X:
    :param tokens:
    :param limit_view:
    """
    tsne = TSNE(n_components=2, random_state=0)

    X = np.stack(list(tokens2vec.values()))
    tokens = list(tokens2vec.keys())

    X_2d = tsne.fit_transform(X)
    colors = range(X_2d.shape[0])

    plt.figure(figsize=(15, 15))
    if tokens is not None:
        for i, token in enumerate(tokens):
            plt.annotate(token, xy=(X_2d[i, 0], X_2d[i, 1]), zorder=1)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=60, alpha=.5)
    plt.title('TSNE visualization of input vectors in 2D')
    # plt.xlabel('x-axis')
    # plt.ylabel('y-axis')
    plt.show()


def cosine_sim(S_vocab, T_vocab, X, limit=1000, save_name='glove'):
    gcn_least_vecs_dict_S, gcn_least_vecs_dict_T = get_freq_disjoint_token_vecs(
        S_vocab, T_vocab, X.detach(), limit=limit, reverse=True)

    gdc_sim_df = heatmap_dict(gcn_least_vecs_dict_S, gcn_least_vecs_dict_T)
    gdc_sim_df.to_csv('gdc_sim_df.csv')

    # plot_vecs_color(gcn_least_vecs_dict_S, gcn_least_vecs_dict_T,
    #                 save_name=save_name + '_tsne.pdf')
    #
    # plot_heatmap(gdc_sim_df, save_name=save_name + '_heat.pdf')


def main():
    """ Main module to start code

    :param args:
        Type: tuple
        Required
        Read Only
    :return:
    """
    torch.load('X_gcn.pt')


if __name__ == "__main__":
    main()
