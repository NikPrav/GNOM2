from File_Handlers.json_handler import read_json


source_vocab = read_json('all_source_cresci15_train_vocab')
target_vocab = read_json('cobo15_train_vocab')
joint_vocab = read_json('joint_cresci15-cobo15_vocab')


import torch


def load_vectors(filename='before.pt'):
    vecs = torch.load(filename, map_location=torch.device('cpu')).detach()
    return vecs


before = load_vectors(filename='before.pt')
after = load_vectors(filename='after.pt')

from sklearn.neighbors import NearestNeighbors

neigh = NearestNeighbors(n_neighbors=5, radius=0.4)
neigh.fit(before)

token_id = 12

kneighbors = neigh.kneighbors(before[token_id].reshape(1, -1), 5, return_distance=False)

for n in kneighbors[0]:
    print(joint_vocab['idx2str_map'][n])
