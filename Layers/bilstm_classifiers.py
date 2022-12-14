# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Short summary of the script.
__description__ : Details and usage.
__project__     : Tweet_Classification
__classes__     : BiLSTM_Classifier
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "07/05/20"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""
import torch
import torch.nn as nn


# class BiLSTM_Emb_Classifier(nn.Module):
#     """ BiLSTM with Embedding layer for classification """
#
#     # define all the layers used in model
#     def __init__(self, vocab_size: int, hid_dim: int, out_dim: int, in_dim: int = 100,
#                  n_layers: int = 2, bidirectional: bool = True, dropout: float = 0.2, num_linear: int = 1) -> None:
#         super(BiLSTM_Emb_Classifier, self).__init__()
#
#         # embedding layer
#         self.embedding = nn.Embedding(vocab_size, in_dim)
#
#         # lstm layer
#         self.lstm = nn.LSTM(in_dim, hid_dim, num_layers=n_layers,
#                             bidirectional=bidirectional, dropout=dropout,
#                             batch_first=True)
#
#         self.linear_layers = []
#         for _ in range(num_linear - 1):
#             if bidirectional:
#                 self.linear_layers.append(nn.Linear(hid_dim * 2, hid_dim * 2))
#             else:
#                 self.linear_layers.append(nn.Linear(hid_dim, hid_dim))
#
#         self.linear_layers = nn.ModuleList(self.linear_layers)
#
#         # Final dense layer
#         if bidirectional:
#             self.fc = nn.Linear(hid_dim * 2, out_dim)
#         else:
#             self.fc = nn.Linear(hid_dim, out_dim)
#
#         # activation function
#         ## NOTE: Sigmoid not required as BCEWithLogitsLoss calculates sigmoid
#         # self.act = nn.Sigmoid()
#
#     def forward(self, text: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
#         """ Takes ids of input text, pads them and predict using BiLSTM.
#
#         Args:
#             text:
#             text_lengths:
#
#         Returns:
#
#         """
#         # text = [batch size,sent_length]
#         embedded = self.embedding(text)
#         # embedded = [batch size, sent_len, emb dim]
#
#         # packed sequence
#         # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded,
#         #                                                     text_lengths,
#         #                                                     batch_first=True)
#
#         # packed_output1, (hidden1, cell) = self.lstm(packed_embedded)
#         embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
#         # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
#         packed_output, (hidden, cell) = self.lstm(embedded)
#         # hidden = [batch size, num num_lstm_layers * num directions, hid dim]
#         # cell = [batch size, num num_lstm_layers * num directions, hid dim]
#
#         # concat the final forward and backward hidden state
#         hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
#
#         for layer in self.linear_layers:
#             hidden = layer(hidden)
#
#         # hidden = [batch size, hid dim * num directions]
#         logits = self.fc(hidden)
#
#         # Final activation function
#         ## NOTE: Sigmoid not required as BCEWithLogitsLoss calculates sigmoid
#         # logits = self.act(logits)
#
#         return logits


class BiLSTM_Emb_Classifier(nn.Module):
    """ BiLSTM with Embedding layer for classification """
    def __init__(self, vocab_size: int, in_dim: int, hid_dim: int, out_dim: int,
                 n_layers: int = 2, bidirectional: bool = True,
                 dropout: float = 0.2, num_linear: int = 1) -> None:
        super(BiLSTM_Emb_Classifier, self).__init__()

        self.bilstm_embedding = BiLSTM_Emb_repr(vocab_size, in_dim, hid_dim,
                                                n_layers, bidirectional, dropout)

        self.linear_layers = []
        for _ in range(num_linear - 1):
            if bidirectional:
                self.linear_layers.append(nn.Linear(hid_dim * 2, hid_dim * 2))
            else:
                self.linear_layers.append(nn.Linear(hid_dim, hid_dim))

        self.linear_layers = nn.ModuleList(self.linear_layers)

        # Final dense layer
        if bidirectional:
            self.fc = nn.Linear(hid_dim * 2, out_dim)
        else:
            self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        """ Takes ids of input text, pads them and predict using BiLSTM.

        Args:
            text:
            text_lengths:

        Returns:

        """
        bilstm_embedded = self.bilstm_embedding(text, text_lengths)

        for layer in self.linear_layers:
            bilstm_embedded = layer(bilstm_embedded)

        # hidden = [batch size, hid dim * num directions]
        logits = self.fc(bilstm_embedded)

        return logits


class BiLSTM_Emb_repr(nn.Module):
    """ BiLSTM with Embedding layer for classification """

    # define all the layers used in model
    def __init__(self, vocab_size: int, in_dim: int, out_dim: int, n_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.2) -> None:
        super(BiLSTM_Emb_repr, self).__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, in_dim)

        # lstm layer
        self.lstm = nn.LSTM(in_dim, out_dim, num_layers=n_layers,
                            bidirectional=bidirectional, dropout=dropout,
                            batch_first=True)

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        """ Takes ids of input text, pads them and predict using BiLSTM.

        Args:
            text:
            text_lengths:

        Returns:

        """
        # text = [batch size,sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]

        # packed_output1, (hidden1, cell) = self.lstm(packed_embedded)
        embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True, enforce_sorted=False)
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        packed_output, (hidden, cell) = self.lstm(embedded)
        # hidden = [batch size, num num_lstm_layers * num directions, hid dim]
        # cell = [batch size, num num_lstm_layers * num directions, hid dim]

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        return hidden


class BiLSTM_Classifier(torch.nn.Module):
    """ BiLSTM for classification (without Embedding layer) """

    # define all the layers used in model
    def __init__(self, in_dim, out_dim, hid_dim=100, n_layers=2,
                 bidirectional=True, dropout=0.2, num_linear=1):
        super(BiLSTM_Classifier, self).__init__()
        self.lstm = torch.nn.LSTM(in_dim, hid_dim, num_layers=n_layers,
                                  bidirectional=bidirectional, dropout=dropout,
                                  batch_first=True)

        ## Intermediate Linear FC layers, default=0
        self.linear_layers = []
        for _ in range(num_linear - 1):
            if bidirectional:
                self.linear_layers.append(torch.nn.Linear(hid_dim * 2, hid_dim * 2))
            else:
                self.linear_layers.append(torch.nn.Linear(hid_dim, hid_dim))

        self.linear_layers = torch.nn.ModuleList(self.linear_layers)

        # Final dense layer
        if bidirectional:
            self.fc = torch.nn.Linear(hid_dim * 2, out_dim)
        else:
            self.fc = torch.nn.Linear(hid_dim, out_dim)

        # activation function
        ## NOTE: Sigmoid not required as BCEWithLogitsLoss calculates sigmoid
        # self.act = torch.nn.Sigmoid()

    def forward(self, text, text_lengths=None):
        """ Takes ids of input text, pads them and predict using BiLSTM.

        Args:
            text: batch size, seq_len, input dim
            text_lengths:

        Returns:

        """
        packed_output, (hidden, cell) = self.lstm(text)
        # hidden = [batch size, num num_lstm_layers * num directions, hid dim]
        # cell = [batch size, num num_lstm_layers * num directions, hid dim]

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        for layer in self.linear_layers:
            hidden = layer(hidden)

        # hidden = [batch size, hid dim * num directions]
        logits = self.fc(hidden)

        # Final activation function
        ## NOTE: Sigmoid not required as BCEWithLogitsLoss calculates sigmoid
        # logits = self.act(logits)

        return logits


class Importance_Estimator(torch.nn.Module):
    """ BiLSTM for classification (without Embedding layer) """

    # define all the layers used in model
    def __init__(self, in_dim, out_dim=1, num_layer=2):
        super(Importance_Estimator, self).__init__()
        self.linear_layers = []
        for _ in range(num_layer - 1):
            self.linear_layers.append(torch.nn.Linear(in_dim, in_dim))

        self.linear_layers = torch.nn.ModuleList(self.linear_layers)
        self.fc = torch.nn.Linear(in_dim, out_dim)

        # self.sig = torch.nn.Sigmoid()
        # self.tanh = torch.nn.Tanh()

    def forward(self, local_input, global_input):
        """ Concatenates local and global vectors to predict importance of global component.

        Args:
            local_input: batch size, input dim
            global_input: batch size, input dim

        Returns:

        """
        combined_input = torch.cat([local_input, global_input], dim=1)
        for layer in self.linear_layers:
            combined_input = layer(combined_input)
        logit = self.fc(combined_input)
        # logit = self.tanh(logit)

        return logit.sigmoid()


def main():
    """
    Main module to start code
    :param args:
        Type: tuple
        Required
        Read Only
    :return:
    """
    pass


if __name__ == "__main__":
    main()
