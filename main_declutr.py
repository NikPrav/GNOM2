from os.path import join
# from declutr import Encoder


from config import cuda_device, emb_dir

# import torch
from torch import LongTensor, no_grad, nn, stack, cat
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import pandas as pd

from Pretrain.pretrain_BERT import get_pretrain_artifacts
from File_Handlers.csv_handler import read_csv, read_csvs
from Text_Processesor.build_corpus_vocab import get_token_embedding
from config import configuration as cfg, platform as plat, username as user, dataset_dir, pretrain_dir
from Metrics.metrics import calculate_performance_bin_sk
from Logger.logger import logger
from Trainer.declutr_lstm_trainer import declutr_trainer


def get_declutr_featurizer():
    # Load the model
    pretrained_model_or_path = join(emb_dir, "declutr-small")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_or_path)
    model = AutoModel.from_pretrained(pretrained_model_or_path)
    model = model.to(cuda_device)

    return tokenizer, model


def get_declutr_text_features(tokenizer, df):
    texts = df.text.to_list()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    for name, tensor in inputs.items():
        inputs[name] = tensor.to(cuda_device)

    return inputs


class declutr_Dataset(Dataset):
    def __init__(self, embs, labels):
        assert embs.size(0) == labels.size(0)
        self.embs = embs
        self.labels = labels

    def __getitem__(self, idx: int):
        """ Get token embedding and label.

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (tensor, list[int])
        """
        return self.embs[idx], self.labels[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.embs)


def embed_text_declutr(tokenizer, model, df,
                       batch_size=cfg['training']["train_batch_size"]):

    inputs = get_declutr_text_features(tokenizer, df)

    embs = []
    # Embed the text
    with no_grad():
        for i in range(0, inputs['input_ids'].size(0), batch_size):
            input_batch = inputs['input_ids'][i:i + batch_size]
            attention_batch = inputs['attention_mask'][i:i + batch_size]
            embs.append(model(input_ids=input_batch,
                              attention_mask=attention_batch,
                              output_hidden_states=False).pooler_output.cpu())

    embs = cat(embs)
    labels = df.labels.to_list()
    labels_pt = LongTensor(labels).unsqueeze(1)
    declutr_dataset = declutr_Dataset(embs, labels_pt)
    declutr_dataloader = DataLoader(declutr_dataset, batch_size, shuffle=True)

    return embs, labels_pt, declutr_dataset, declutr_dataloader


def replace_init_embs(model, tokenizer, pepoch=cfg['pretrain']['epoch']) -> None:
    """ Replace declutr input tokens embeddings with pretrained embeddings.

    :param model: simpletransformer model
    :param embs_dict: Dict of token to emb (Pytorch Tensor).
    """
    logger.info('Fatching model (declutr) init embs')
    orig_embs = model.embeddings.word_embeddings.weight.cpu()
    logger.info((orig_embs, orig_embs.shape))
    orig_embs_dict = {}
    logger.info('Create token2emb map')
    for token, idx in tokenizer.vocab.items():
        orig_embs_dict[token] = orig_embs[idx]
    token_list = list(tokenizer.vocab.keys())

    logger.info('Pre-train (declutr) init embs using GCN')
    joint_vocab, token2pretrained_embs, X = get_pretrain_artifacts(orig_embs_dict, epoch=pepoch)

    logger.info('Get pretrained (declutr) embs')
    embs, _ = get_token_embedding(token_list, oov_embs=token2pretrained_embs,
                                  default_embs=orig_embs_dict, add_unk=False)
    embs = nn.Parameter(embs.to(cuda_device))
    logger.info('Reassign (declutr) embs to model')
    model.embeddings.word_embeddings.weight = embs
    logger.info((model.embeddings.word_embeddings.weight,
                 model.embeddings.word_embeddings.weight.shape))


def main():
    if cfg['data']['zeroshot']:
        train_df = read_csvs(data_dir=pretrain_dir, filenames=cfg['pretrain']['files'])
        train_df = train_df.sample(frac=1)
        val_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['val'])
        val_df["labels"] = pd.to_numeric(val_df["labels"], downcast="float")
        test_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['test'])
        test_df = test_df.sample(frac=1)
        test_df["labels"] = pd.to_numeric(test_df["labels"], downcast="float")
    else:
        train_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['name'])
        train_df = train_df.sample(frac=1)
        train_df["labels"] = pd.to_numeric(train_df["labels"], downcast="float")
        val_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['val'])
        val_df["labels"] = pd.to_numeric(val_df["labels"], downcast="float")
        test_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['test'])
        test_df = test_df.sample(frac=1)
        test_df["labels"] = pd.to_numeric(test_df["labels"], downcast="float")

    tokenizer, model = get_declutr_featurizer()

    replace_init_embs(model, tokenizer, pepoch=1)

    train_embs, train_labels_pt, train_dataset, train_dataloader, = embed_text_declutr(tokenizer, model, train_df)
    test_embs, test_labels_pt, test_dataset, test_dataloader, = embed_text_declutr(tokenizer, model, test_df)
    val_embs, val_labels_pt, val_dataset, val_dataloader, = embed_text_declutr(tokenizer, model, val_df)

    model.to('cpu')

    declutr_trainer(
        train_dataloader, val_dataloader, test_dataloader, in_dim=768, hid_dim=100,
        epoch=cfg['training']['num_epoch'], loss_func=nn.BCEWithLogitsLoss(),
        lr=cfg["model"]["optimizer"]["lr"], model_name='declutr')


if __name__ == "__main__":
    main()
