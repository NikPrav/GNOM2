import argparse
import torch
import torch.optim as optim
import torchtext as tx
from collections import OrderedDict
from json import dumps

from Layers.bilstm_classifiers import BiLSTM_Classifier
from Text_Processesor.build_corpus_vocab import get_dataset_fields
from Utils.utils import split_target, logit2label
from Metrics.metrics import calculate_performance_pl as calculate_performance
from config import configuration as cfg, dataset_dir
from Logger.logger import logger


def training(model, dataloader: torch.utils.data.dataloader.DataLoader,
             loss_func: torch.nn.modules.loss.BCEWithLogitsLoss,
             optimizer, epoch=5,
             eval_dataloader: torch.utils.data.dataloader.DataLoader = None):
    train_epoch_losses = []
    train_epoch_dict = OrderedDict()
    for epoch in range(epoch):
        model.train()
        epoch_loss = 0
        preds = []
        trues = []
        for iter, (graph_batch, label) in enumerate(dataloader):
            ## Store emb in a separate file as self_loop removes emb info:
            emb = graph_batch.ndata['emb']
            # graph_batch = dgl.add_self_loop(graph_batch)
            prediction = model(graph_batch, emb)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            preds.append(prediction.detach())
            trues.append(label.detach())
        epoch_loss /= (iter + 1)
        losses, test_output = testing(model, loss_func=loss_func, dataloader=eval_dataloader)
        logger.info(f"Epoch {epoch}, Train loss {epoch_loss}, Eval loss {losses},"
                    f" Weighted F1 {test_output['result']['f1_weighted'].item()}")
        # logger.info(dumps(test_output['result'], indent=4))
        train_epoch_losses.append(epoch_loss)
        preds = torch.cat(preds)

        ## Converting raw scores to probabilities using Sigmoid:
        preds = torch.sigmoid(preds)

        ## Converting probabilities to class labels:
        preds = logit2label(preds.detach(), cls_thresh=0.5)
        trues = torch.cat(trues)
        result_dict = calculate_performance(trues, preds)
        # logger.info(dumps(result_dict, indent=4))
        train_epoch_dict[epoch] = {
            'preds':  preds,
            'trues':  trues,
            'result': result_dict
        }
        # logger.info(f'Epoch {epoch} result: \n{result_dict}')

    return train_epoch_losses, train_epoch_dict


def testing(model, loss_func, dataloader: torch.utils.data.dataloader.DataLoader):
    model.eval()
    preds = []
    trues = []
    losses = []
    for iter, (graph_batch, label) in enumerate(dataloader):
        ## Store emb in a separate file as self_loop removes emb info:
        emb = graph_batch.ndata['emb']
        # graph_batch = dgl.add_self_loop(graph_batch)
        prediction = model(graph_batch, emb)
        loss = loss_func(prediction, label)
        preds.append(prediction.detach())
        trues.append(label.detach())
        losses.append(loss.detach())
    losses = torch.mean(torch.stack(losses))
    preds = torch.cat(preds)

    ## Converting raw scores to probabilities using Sigmoid:
    preds = torch.sigmoid(preds)

    ## Converting probabilities to class labels:
    preds = logit2label(preds.detach(), cls_thresh=0.5)
    trues = torch.cat(trues)
    result_dict = calculate_performance(trues, preds)
    test_output = {
        'preds':  preds,
        'trues':  trues,
        'result': result_dict
    }
    # logger.info(dumps(result_dict, indent=4))

    return losses, test_output


def multilabel_classifier(data, epoch=cfg['training']['num_epoch'], n_classes=cfg['data']['num_classes']):
    model = BiLSTM_Classifier(vocab_size, hid_dim=50, out_dim=n_classes,
                              embedding_dim=cfg['embeddings']['emb_dim'],
                              n_layers=2,
                              bidirectional=True,
                              dropout=0.2, num_linear=1)
    logger.info(model)

    # loss_func = torch.nn.CrossEntropyLoss()
    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["model"]["optimizer"]["lr"])

    epoch_losses, train_epochs_output_dict = training(
        model, data.train_dataloader(), loss_func=loss_func, optimizer=optimizer,
        epoch=epoch, eval_dataloader=data.test_dataloader())

    losses, test_output = testing(model, loss_func=loss_func, dataloader=data.test_dataloader())
    logger.info(dumps(test_output['result'], indent=4))

    return train_epochs_output_dict, test_output


class DataFrameDataset(tx.data.Dataset):

    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            label = row.target if not is_test else None
            text = row.text
            examples.append(torch.data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, True, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


if __name__ == "__main__":
    from Data_Handlers.torchtext_handler import dataset2bucket_iter
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("-d", "--dataset_name",
                        default=cfg['data']['source']['labelled'], type=str)
    parser.add_argument("-m", "--model_name",
                        default=cfg['transformer']['model_name'], type=str)
    parser.add_argument("-mt", "--model_type",
                        default=cfg['transformer']['model_type'], type=str)
    parser.add_argument("-ne", "--num_train_epochs",
                        default=cfg['training']['num_epoch'], type=int)
    parser.add_argument("-c", "--use_cuda",
                        default=cfg['cuda']['use_cuda'], action='store_true')

    args = parser.parse_args()

    data_dir = dataset_dir

    from File_Handlers.read_datasets import load_fire16, load_smerp17

    if args.dataset_name.startswith('fire16'):
        train_df = load_fire16()

    if cfg['data']['target']['labelled'].startswith('smerp17'):
        target_df = load_smerp17()
        target_train_df, test_df = split_target(df=target_df, test_size=0.4)

    target_train_df.to_csv('train_a')
    test_df.to_csv('test_a')

    train_dataset, (train_fields, LABEL) = get_dataset_fields(csv_dir='', csv_file='train_a')
    test_dataset, (TEXT, LABEL) = get_dataset_fields(csv_dir='', csv_file='test_a')

    train_iter, val_iter = dataset2bucket_iter(
        (train_dataset, test_dataset), batch_sizes=(32, 64))

    size_of_vocab = len(train_fields.vocab)
    num_output_nodes = 4

    multilabel_classifier(target_train_df, test_df)
