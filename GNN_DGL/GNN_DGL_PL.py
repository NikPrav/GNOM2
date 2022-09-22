# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : PyTorch Lightning class for graph and node classification
__description__ : Graph and Node classification using DGL library
__project__     : Tweet_GNN_inductive
__classes__     : Tweet_GNN_inductive
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "05/08/20"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

import json
import pathlib
import torch
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
# from pytorch_lightning.metrics.sklearns import F1
from dgl import mean_nodes, DGLGraph, graph
from dgl.nn.pytorch.conv import GATConv

from config import configuration as cfg
from Logger.logger import logger


class GAT_Graph_Classifier(pl.LightningModule):
    """
    GAT model class: This is where the learning happens
    The boilerplate for learning is abstracted away by Lightning
    """
    def __init__(self, in_dim, hid_dim, num_heads, n_classes):
        super(GAT_Graph_Classifier, self).__init__()
        self.conv1 = GATConv(in_dim, hid_dim, num_heads)
        self.conv2 = GATConv(hid_dim * num_heads, hid_dim, num_heads)
        self.classify = torch.nn.Linear(hid_dim * num_heads, n_classes)

    def forward(self, g, emb=None):
        if emb is None:
            # Use node degree as the initial node feature.
            # For undirected graphs, the in-degree is the
            # same as the out_degree.
            # emb = g.in_degrees().view(-1, 1).float()

            ## Extract node features from graph.ndata
            emb = g.ndata['emb']

        # Perform graph convolution and activation function.
        emb = F.relu(self.conv1(g, emb))
        emb = emb.view(-1, emb.size(1) * emb.size(2)).float()
        emb = F.relu(self.conv2(g, emb))
        emb = emb.view(-1, emb.size(1) * emb.size(2)).float()
        g.ndata['emb'] = emb

        # Calculate graph representation by averaging all node representations.
        hg = mean_nodes(g, 'emb')
        return self.classify(hg)

    def configure_optimizers(self, lr=cfg['training']['optimizer']['lr']):
        return optim.Adam(self.parameters(), lr=lr)

    def loss_function(self, prediction, label):
        return F.binary_cross_entropy_with_logits(prediction, label)

    def shared_step(self, batch):
        """ Step shared among train, validation and test.

        :param batch: Tuple of a batch of graphs as single large graph and labels.
        :return:
        """
        graph_batch, labels = batch
        # labels = torch.Tensor(labels)
        prediction = self(graph_batch)
        labels = labels.type_as(prediction)
        loss = self.loss_function(prediction, labels)
        return loss, prediction

    def _calc_metrics(self, outputs):
        """
        helper function to calculate the metrics
        """
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        predictions = torch.Tensor()
        # labels = torch.Tensor()
        labels = torch.LongTensor()
        for x in outputs:
            predictions = torch.cat((predictions, x['prediction']), 0)
            labels = torch.cat((labels, x['labels']), 0)
        # self.logger.experiment._
        labels = torch.Tensor(labels)
        metric = F1(average='macro')
        avg_f1_score = sum(list(map(lambda pred, y: metric(pred > 0.5, y),
                                    predictions, labels)))/predictions.shape[0]

        class_f1_scores_list = torch.Tensor([0 for i in range(len(predictions[0]))])
        for i, pred in enumerate(predictions):
            label_list = labels[i]
            for j, class_pred in enumerate(pred):
                class_f1_scores_list[j] += metric(class_pred.item() > 0.5,
                                                  label_list[j].item()).item()
        class_f1_scores_list /= len(predictions)
        class_f1_scores = {}
        try:
            label_text_to_label_id_path = cfg['paths']['data_root'] + cfg['paths']['label_text_to_label_id']
            assert pathlib.Path(label_text_to_label_id_path).exists(), \
                "Label to id path is not valid! Using Incremental class names"
            with open(label_text_to_label_id_path, "r") as f:
                label_text_to_label_id = json.load(f)
            label_id_to_label_text = {value: key for key, value in label_text_to_label_id.items()}

        except Exception:
            label_id_to_label_text = {i: f'class_{i}' for i in range(len(class_f1_scores_list))}

        for index in range(len(label_id_to_label_text)):
            f1_score = class_f1_scores_list[index]
            class_name = label_id_to_label_text[index]
            class_f1_scores[class_name] = f1_score
        return avg_loss, avg_f1_score, class_f1_scores

    def training_step(self, batch, batch_idx):
        """ Training 1 iteration

        :param batch:
        :param batch_idx:
        :return:
        """
        graph_batch, labels = batch
        batch_train_loss, prediction = self.shared_step(batch)
        prediction = torch.sigmoid(prediction)
        return {'loss': batch_train_loss, 'prediction': prediction, 'labels': labels}

    def train_epoch_end(self, outputs):
        """ Training epoch end operation.

        :param outputs:
        :return:
        """
        avg_train_loss, avg_f1_score, class_f1_scores = self._calc_metrics(outputs)

        logger.info(f"Train class f1 scores {class_f1_scores}")
        log = {'avg_train_loss': avg_train_loss, 'avg_train_f1_score': avg_f1_score}
        return {'log': log}

    def validation_step(self, batch, batch_idx):
        """ 1 Validation operation

        :param batch:
        :param batch_idx:
        :return:
        """
        graph_batch, labels = batch
        batch_val_loss, prediction = self.shared_step(batch)
        prediction = torch.sigmoid(prediction)
        # result = pl.TrainResult(val_loss)
        # result.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': batch_val_loss, 'prediction': prediction, 'labels': labels}

    def validation_epoch_end(self, outputs):
        """ Tasks should be performed after validation epoch.

        :param outputs:
        :return:
        """
        avg_val_loss, avg_f1_score, class_f1_scores = self._calc_metrics(outputs)
        logger.info(f"Train class f1 scores {class_f1_scores}")
        log = {'avg_val_loss': avg_val_loss, 'avg_val_f1_score': avg_f1_score}
        return {'log': log}

    def test_step(self, batch, batch_idx):
        graph_batch, labels = batch
        batch_test_loss, prediction = self.shared_step(batch)
        prediction = torch.sigmoid(prediction)
        return {'loss': batch_test_loss, 'prediction': prediction, 'labels': labels}

    def test_epoch_end(self, outputs):
        avg_test_loss, avg_f1_score, class_f1_scores = self._calc_metrics(outputs)
        logger.info(f"Train class f1 scores {class_f1_scores}")
        log = {'avg_test_loss': avg_test_loss, 'avg_test_f1_score': avg_f1_score}
        return {'log': log}
        # TODO tensorboard for logging metrics


class GAT_Node_Classifier(pl.LightningModule):
    """
    GAT model class: This is where the learning happens
    The boilerplate for learning is abstracted away by Lightning
    """
    def __init__(self, in_dim, hid_dim, num_heads, out_dim):
        super(GAT_Node_Classifier, self).__init__()

        self.layer1 = GATConv(in_dim, hid_dim, num_heads)
        # Be aware that the input dimension is hid_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATConv(hid_dim * num_heads, out_dim, 1)

    def forward(self, g: DGLGraph, emb: torch.Tensor) -> torch.Tensor:
        emb = self.layer1(g, emb)
        ## Concatenating multiple head embeddings
        emb = emb.view(-1, emb.size(1) * emb.size(2))
        emb = F.elu(emb)
        emb = self.layer2(g, emb).squeeze()
        return emb

    def configure_optimizers(self, lr=cfg['training']['optimizer']['lr']):
        return optim.Adam(self.parameters(), lr=lr)

    def loss_function(self, prediction, label):
        return F.binary_cross_entropy_with_logits(prediction, label)

    def shared_step(self, batch):
        """ Step shared among train, validation and test.

        :param batch: Tuple of a batch of graphs as single large graph and labels.
        :return:
        """
        graph_batch, labels = batch
        labels = torch.Tensor(labels)
        prediction = self(graph_batch)
        labels = labels.type_as(prediction)
        loss = self.loss_function(prediction, labels)
        return loss, prediction

    def _calc_metrics(self, outputs):
        """
        helper function to calculate the metrics
        """
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        predictions = torch.Tensor()
        # labels = torch.Tensor()
        labels = torch.LongTensor()
        for x in outputs:
            predictions = torch.cat((predictions, x['prediction']), 0)
            labels = torch.cat((labels, x['labels']), 0)
        # self.logger.experiment._
        labels = torch.Tensor(labels)
        # metric = F1(average='macro')
        avg_f1_score = sum(list(map(lambda pred, y: metric(pred > 0.5, y),
                                    predictions, labels)))/predictions.shape[0]

        class_f1_scores_list = torch.Tensor([0 for i in range(len(predictions[0]))])
        for i, pred in enumerate(predictions):
            label_list = labels[i]
            for j, class_pred in enumerate(pred):
                class_f1_scores_list[j] += metric(class_pred.item() > 0.5,
                                                  label_list[j].item()).item()
        class_f1_scores_list /= len(predictions)
        class_f1_scores = {}
        try:
            label_text_to_label_id_path = cfg['paths']['data_root'] + cfg['paths']['label_text_to_label_id']
            assert pathlib.Path(label_text_to_label_id_path).exists(),\
                "Label to id path is not valid! Using Incremental class names"
            with open(label_text_to_label_id_path, "r") as f:
                label_text_to_label_id = json.load(f)
            label_id_to_label_text = {value: key for key, value in label_text_to_label_id.items()}

        except Exception:
            label_id_to_label_text = {i: f'class_{i}' for i in range(len(class_f1_scores_list))}

        for index in range(len(label_id_to_label_text)):
            f1_score = class_f1_scores_list[index]
            class_name = label_id_to_label_text[index]
            class_f1_scores[class_name] = f1_score
        return avg_loss, avg_f1_score, class_f1_scores

    def training_step(self, batch, batch_idx):
        """ Training 1 iteration

        :param batch:
        :param batch_idx:
        :return:
        """
        graph_batch, labels = batch
        batch_train_loss, prediction = self.shared_step(batch)
        prediction = torch.sigmoid(prediction)
        return {'loss': batch_train_loss, 'prediction': prediction, 'labels': labels}

    def train_epoch_end(self, outputs):
        """ Training epoch end operation.

        :param outputs:
        :return:
        """
        avg_train_loss, avg_f1_score, class_f1_scores = self._calc_metrics(outputs)

        logger.info(f"Train class f1 scores {class_f1_scores}")
        log = {'avg_train_loss': avg_train_loss, 'avg_train_f1_score': avg_f1_score}
        return {'log': log}

    def validation_step(self, batch, batch_idx):
        """ 1 Validation operation

        :param batch:
        :param batch_idx:
        :return:
        """
        graph_batch, labels = batch
        batch_val_loss, prediction = self.shared_step(batch)
        prediction = torch.sigmoid(prediction)
        # result = pl.TrainResult(val_loss)
        # result.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': batch_val_loss, 'prediction': prediction, 'labels': labels}

    def validation_epoch_end(self, outputs):
        """ Tasks should be performed after validation epoch.

        :param outputs:
        :return:
        """
        avg_val_loss, avg_f1_score, class_f1_scores = self._calc_metrics(outputs)
        logger.info(f"Train class f1 scores {class_f1_scores}")
        log = {'avg_val_loss': avg_val_loss, 'avg_val_f1_score': avg_f1_score}
        return {'log': log}

    def test_step(self, batch, batch_idx):
        graph_batch, labels = batch
        batch_test_loss, prediction = self.shared_step(batch)
        prediction = torch.sigmoid(prediction)
        return {'loss': batch_test_loss, 'prediction': prediction, 'labels': labels}

    def test_epoch_end(self, outputs):
        avg_test_loss, avg_f1_score, class_f1_scores = self._calc_metrics(outputs)
        logger.info(f"Train class f1 scores {class_f1_scores}")
        log = {'avg_test_loss': avg_test_loss, 'avg_test_f1_score': avg_f1_score}
        return {'log': log}
        # TODO tensorboard for logging metrics
