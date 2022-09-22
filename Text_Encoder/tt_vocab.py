from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, fromstring
from gensim import utils
import gensim
import numpy as np


def my_save_word2vec_format(fname, vocab, vectors, binary=True, total_vec=2):
    """Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.

    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : dict
        The vocabulary of words.
    vectors : numpy.array
        The vectors to be stored.
    binary : bool, optional
        If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
    total_vec : int, optional
        Explicitly specify total number of vectors
        (in case word vectors are appended with document vectors afterwards).

    """
    if not (vocab or vectors):
        raise RuntimeError("no input")
    if total_vec is None:
        total_vec = len(vocab)
    vector_size = vectors.shape[1]
    assert (len(vocab), vector_size) == vectors.shape
    with utils.smart_open(fname, 'wb') as fout:
        print(total_vec, vector_size)
        fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in vocab.items():
            if binary:
                row = row.astype(REAL)
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))


d = {'1': np.random.randn(300), '2': np.random.randn(300)}

m = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=300)
m.vocab = d
m.vectors = np.array(list(d.values()))
my_save_word2vec_format(binary=True, fname='train.bin', total_vec=len(d), vocab=m.vocab, vectors=m.vectors)
