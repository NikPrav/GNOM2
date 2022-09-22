from json import load, loads, dumps
from collections import OrderedDict
from pathlib import Path

def read_json(file_path: str, convert_ordereddict=True) -> OrderedDict:
    """ Reads json file as OrderedDict.

    :param convert_ordereddict:
    :param file_path:
    :return:
    """
    file_path = Path(file_path + ".json")
    # file_path = Path(file_path)
    print(f"Reading json file [{file_path}].")

    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as file:
            data = load(file)
            if convert_ordereddict:
                data = OrderedDict(data)
        return data
    else:
        raise FileNotFoundError("File [{}] not found.".format(file_path))


source_vocab = read_json('all_source_cresci15_train_vocab')
target_vocab = read_json('cobo15_train_vocab')
joint_vocab = read_json('joint_cresci15-cobo15_vocab')

import random


def select_tokens(source_vocab, target_vocab, token_count=25, select_mode='exclusive'):
    if select_mode=='exclusive':
        source_tokens = set(source_vocab['idx2str_map']) - set(target_vocab['idx2str_map'])
        target_tokens = set(target_vocab['idx2str_map']) - set(source_vocab['idx2str_map'])
        print(len(source_vocab['idx2str_map']), len(source_tokens), 
            len(target_vocab['idx2str_map']), len(target_tokens))
    else: # Intersection
        source_tokens = target_tokens = set(source_vocab['idx2str_map']
            ).intersection(set(target_vocab['idx2str_map']))
    
    source_selected = random.sample(source_tokens, token_count)
    target_selected = random.sample(target_tokens, token_count)

    tokens = source_selected + target_selected

    return tokens, source_selected, target_selected

def get_ids(vocab, words):
    idxs = {}
    for word in words:
        idxs[word] = vocab['str2idx_map'].get(word, 0)

    return idxs

tokens, source_tokens_selected, target_tokens_selected = select_tokens(
    source_vocab, target_vocab, token_count=25, select_mode='exclusive')

source_tokens_idx = get_ids(joint_vocab, source_tokens_selected)
target_tokens_idx = get_ids(joint_vocab, target_tokens_selected)

import torch

def load_vectors(filename='before.pt'):
    vecs = torch.load(filename, map_location=torch.device('cpu')).detach()
    return vecs

before = load_vectors(filename='before.pt')

after = load_vectors(filename='after.pt')

def get_token_vectors(source_tokens_idx, target_tokens_idx):
    before_vecs = torch.concat((before[list(source_tokens_idx.values())], before[list(target_tokens_idx.values())]), dim=0)
    after_vecs = torch.concat((after[list(source_tokens_idx.values())], after[list(target_tokens_idx.values())]), dim=0)

    return before_vecs, after_vecs

before_vecs, after_vecs = get_token_vectors(source_tokens_idx, target_tokens_idx)


!pip install umap-learn
from umap import UMAP

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

def plot_features_umap(X, tokens: list = None, limit_view: int = -100,
                       name='fire-smerp_umap.pdf'):
    """ Plots TSNE representations of tokens and their embeddings.

    :param X:
    :param tokens:
    :param limit_view:
    """
    umap = UMAP(n_components=2, random_state=0)

    if limit_view > X.shape[0]:
        limit_view = X.shape[0]

    if limit_view > 0:
        X = X[:limit_view, ]
        if tokens is not None:
            tokens = tokens[:limit_view]
    elif limit_view < 0:
        X = X[limit_view:, ]
        if tokens is not None:
            tokens = tokens[limit_view:]
    else:
        pass

    X_2d = umap.fit_transform(X)
    colors = [1] * (X_2d.shape[0] // 2) + [2] * (X_2d.shape[0] // 2)
    print(colors)

    # plt.figure(figsize=(6, 5))
    
    fig, ax = plt.subplots()
    plt.tight_layout()

    ax.set_xticks([])
    ax.set_yticks([])
    
    if tokens is not None:
        for i, token in enumerate(tokens):
            plt.annotate(token, xy=(X_2d[i, 0], X_2d[i, 1]), zorder=1)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=60, alpha=.5)
    # plt.title('UMAP visualization of input vectors in 2D')
    # plt.xlabel('x-axis')
    # plt.ylabel('y-axis')
    plt.show()

    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0]*2, Size[1]*2, forward=True)
    fig.savefig(name, format='pdf', bbox_inches='tight')


plot_features_umap(X=before_vecs,
                #    tokens=tokens
                   name='UMAP_before_fire-smerp.pdf'
                   )


plot_features_umap(X=after_vecs,
                #    tokens=tokens
                   name='UMAP_after_fire-smerp.pdf'
                   )

