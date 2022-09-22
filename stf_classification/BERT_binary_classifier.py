# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Generate token graph
__description__ : Details and usage.
__project__     : Tweet_GNN_inductive
__classes__     : Tweet_GNN_inductive
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
import timeit
import argparse
import numpy as np
import pandas as pd
from os.path import join
from os import environ
from json import dumps, dump
from simpletransformers.classification import MultiLabelClassificationModel, MultiLabelClassificationArgs, ClassificationModel, ClassificationArgs
# from simpletransformers.language_representation import RepresentationModel
# from simpletransformers.config.model_args import ModelArgs

from File_Handlers.csv_handler import read_csv, read_csvs
from Text_Processesor.build_corpus_vocab import get_token_embedding
from config import configuration as cfg, platform as plat, username as user, dataset_dir, pretrain_dir
from Metrics.metrics import calculate_performance_bin_sk
from Logger.logger import logger


if torch.cuda.is_available():
    # environ["CUDA_VISIBLE_DEVICES"] = str(cfg['cuda']['cuda_devices'][plat][user])
    environ["CUDA_VISIBLE_DEVICES"] = '1'
    torch.cuda.set_device(cfg['cuda']['cuda_devices'][plat][user])


def format_inputs(df: pd.core.frame.DataFrame):
    """ Converts the input to proper format for simpletransformer.

    """
    df['labels'] = df[df.columns[1:]].values.tolist()
    df = df[['text', 'labels']].copy()
    return df


def macro_f1(labels, preds, threshold=0.5):
    """ Converts probabilities to labels

     using the [threshold] and calculates metrics.

    Parameters
    ----------
    labels
    preds
    threshold

    Returns
    -------

    """
    np.savetxt(join(cfg['paths']['dataset_root'][plat][user],
                    cfg['data']['name'] + "_" +
                    cfg['transformer']['model_type'] + '_labels.txt'),
               labels)
    np.savetxt(join(cfg['paths']['dataset_root'][plat][user],
                    cfg['data']['name'] + "_" +
                    cfg['transformer']['model_type'] + '_preds.txt'),
               preds)

    preds[preds > threshold] = 1
    preds[preds <= threshold] = 0

    scores = calculate_performance_bin_sk(labels, preds)
    scores['dataset'] = cfg['data']['name']
    scores['epoch'] = cfg['training']['num_epoch']
    logger.info(f"Scores: [{threshold}]:\n[{dumps(scores, indent=4)}]")
    logger.info(f"Epoch {scores['epoch']} Test W-F1"
                f" {scores['f1_weighted'].item():1.4}")
    return scores['f1_weighted']


def replace_bert_init_embs(model: ClassificationModel, embs_dict: dict) -> None:
    """ Replace bert input tokens embeddings with custom embeddings.

    :param model: simpletransformer model
    :param embs_dict: Dict of token to emb (Pytorch Tensor).
    """
    orig_embs = model.model.bert.embeddings.word_embeddings.weight
    orig_embs_dict = {}
    for token, idx in model.tokenizer.vocab.items():
        orig_embs_dict[token] = orig_embs[idx]
    token_list = list(model.tokenizer.vocab.keys())
    embs, _ = get_token_embedding(token_list, oov_embs=embs_dict,
                                  default_embs=orig_embs_dict)
    embs = torch.nn.Parameter(embs)
    model.model.bert.embeddings.word_embeddings.weight = embs
    # embs = torch.nn.Embedding(embs)
    # model.model.bert.set_input_embeddings(embs)


def BERT_binary_classifier(train_df=None, test_df=None, epoch=cfg['training']['num_epoch'], train_all_bert=True):
    # if cfg['data']['zeroshot']:
    #     train_df = read_csvs(data_dir=pretrain_dir, filenames=cfg['pretrain']['files'])
    #     train_df = train_df.sample(frac=1)
    #     test_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['test'])
    #     test_df = test_df.sample(frac=1)
    #     test_df["labels"] = pd.to_numeric(test_df["labels"], downcast="float")
    # else:
    #     train_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['name'])
    #     train_df = train_df.sample(frac=1)
    #     train_df["labels"] = pd.to_numeric(train_df["labels"], downcast="float")
    #     val_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['val'])
    #     val_df["labels"] = pd.to_numeric(val_df["labels"], downcast="float")
    #     test_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['test'])
    #     test_df = test_df.sample(frac=1)
    #     test_df["labels"] = pd.to_numeric(test_df["labels"], downcast="float")

    # Optional model configuration
    model_args = ClassificationArgs(
        num_train_epochs=epoch, train_batch_size=64, eval_batch_size=128,
        overwrite_output_dir=True, evaluate_during_training=True,
        evaluate_during_training_verbose=True, evaluate_during_training_silent=False)
    model_args.overwrite_output_dir = True
    model_args.evaluate_during_training_steps = 870
    model_args.evaluate_each_epoch = True
    # model_args.early_stopping_consider_epochs = True
    model_args.use_early_stopping = True
    model_args.save_model_every_epoch = False
    model_args.save_eval_checkpoints = False
    model_args.save_optimizer_and_scheduler = False
    if not train_all_bert:
        logger.warning(f'Training classifier only.')
        model_args.train_custom_parameters_only = True
        model_args.custom_parameter_groups = [
            {
                "params": ["classifier.weight"],
                "lr":     1e-3,
            },
            {
                "params":       ["classifier.bias"],
                "lr":           1e-3,
                "weight_decay": 0.0,
            },
        ]

    # Create a ClassificationModel
    model = ClassificationModel("bert", "bert-base-uncased", args=model_args)

    # replace_bert_init_embs(model, embs_dict=None)

    # Train the model
    model.train_model(train_df, eval_df=test_df, verbose=True, macro_f1=macro_f1)

    # # Evaluate the model
    # if cfg['data']['zeroshot']:
    #     for test_data in cfg['data']['all_test']:
    #         logger.info(f'TEST data: {test_data}')
    #         test_df = read_csv(data_dir=pretrain_dir, data_file=test_data)
    #         test_df = test_df.sample(frac=1)
    #         test_df["labels"] = pd.to_numeric(test_df["labels"], downcast="float")
    #         _, _, _ = model.eval_model(test_df, weighted_f1=weighted_f1)
    # else:
    #     result, model_outputs, wrong_predictions = model.eval_model(test_df, weighted_f1=weighted_f1)

    # Make predictions with the model
    # predictions, raw_outputs = model.predict(["Sam was a Wizard"])


if __name__ == "__main__":
    BERT_binary_classifier()

    # parser = argparse.ArgumentParser()
    #
    # ## Required parameters
    # parser.add_argument("-d", "--dataset_name",
    #                     default=cfg['data']['name'], type=str)
    # parser.add_argument("-m", "--model_name",
    #                     default=cfg['transformer']['model_name'], type=str)
    # parser.add_argument("-mt", "--model_type",
    #                     default=cfg['transformer']['model_type'], type=str)
    # parser.add_argument("-ne", "--num_train_epochs",
    #                     default=cfg['training']['num_epoch'], type=int)
    # parser.add_argument("-c", "--use_cuda",
    #                     default=cfg['cuda']['use_cuda'], action='store_true')
    #
    # args = parser.parse_args()
    #
    # train_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['name'])
    # train_df = train_df.sample(frac=1)
    # train_df["labels"] = pd.to_numeric(train_df["labels"], downcast="float")
    # val_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['val'])
    # val_df["labels"] = pd.to_numeric(val_df["labels"], downcast="float")
    # test_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['test'])
    # test_df = test_df.sample(frac=1)
    # test_df["labels"] = pd.to_numeric(test_df["labels"], downcast="float")
    #
    # result, model_outputs = BERT_classifier(
    #     train_df=train_df, test_df=test_df, dataset_name=args.dataset_name,
    #     model_name=args.model_name, model_type=args.model_type,
    #     num_epoch=args.num_train_epochs, use_cuda=args.use_cuda)
