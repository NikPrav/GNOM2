# coding=utf-8
# !/usr/bin/python3.6  # Please use python 3.6
"""
__synopsis__    : Short summary of the script.
__description__ : Details and usage.
__project__     : Tweet_GNN_inductive
__classes__     : main_stf
__variables__   :
__methods__     :
__author__      : User
__version__     : ":  "
__date__        : "01-12-2021"
__last_modified__:
__copyright__   : "Copyright (c) 2021, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license found in the LICENSE file in the root
                   directory of this source tree."
"""
import ast, random
import pandas as pd
from json import dumps
from copy import deepcopy
from os import environ, makedirs, listdir
from os.path import join, exists
from pathlib import Path
from torch import cuda, save, load, nn, no_grad, tensor, long, set_num_threads,\
    manual_seed, device, Tensor, qint8, quantization, mean, cat, stack, mm,\
    triu, add, argmax, softmax as t_softmax
from config import configuration as cfg, platform as plat, username as user,\
    dataset_dir, pretrain_dir, cuda_device

# from Text_Processesor.build_corpus_vocab import get_dataset_fields
from Graph_Construction.create_subword_token_graph import construct_token_graph
from Metrics.metrics import weighted_f1
from File_Handlers.csv_handler import read_csv, read_csvs
from Utils.utils import set_all_seeds
from Logger.logger import logger

from stf_config.model_args import ClassificationArgs
from stf_classification.classification_model import ClassificationModel


def format_df_cls(df: pd.core.frame.DataFrame):
    """ Converts input to proper format for simpletransformer. """
    ## Consolidate labels:
    df['labels'] = df[df.columns[1:]].values.tolist()
    ## Keep required columns only:
    df = df[['text', 'labels']].copy()
    return df


def read_data(train_name, val_name, test_name, data_dir=pretrain_dir,
              format_input=True, zeroshot=cfg['data']['zeroshot'],
              zeroshot_train_names=cfg['pretrain']['files'],
              train_portion=None):
    if zeroshot:
        train_df = read_csvs(data_dir=data_dir, filenames=zeroshot_train_names)
        train_df = train_df.sample(frac=1)
        logger.warning(f"Zero-Shot train size {train_df.shape} from {zeroshot_train_names}")
    else:
        train_df = read_csv(data_dir=data_dir, data_file=train_name)
        train_df = train_df.sample(frac=1)

    ## Reduce train size
    if train_portion:
        train_df = train_df.sample(frac=train_portion)
        logger.warning(f'Reducing train portion to {args.train_portion}'
                       f' with {train_df.shape}')
    val_df = read_csv(data_dir=data_dir, data_file=val_name)
    val_df = val_df.sample(frac=1)
    test_df = read_csv(data_dir=data_dir, data_file=test_name)
    test_df = test_df.sample(frac=1)
    # test_df["labels"] = pd.to_numeric(test_df["labels"], downcast="float")

    logger.info(f'Data shapes:\nTrain {train_df.shape},\nVal {val_df.shape},'
                f'\nTest {test_df.shape}')

    try:
        ## Check if labels are in str format:
        if type(train_df.labels.iloc[0]) == str:
            train_df['labels'] = train_df['labels'].map(ast.literal_eval)
            val_df['labels'] = val_df['labels'].map(ast.literal_eval)
            test_df['labels'] = test_df['labels'].map(ast.literal_eval)
    except AttributeError as e:
        logger.info(f"Formatting data for simpletransformers.")
        train_df = format_df_cls(train_df)
        val_df = format_df_cls(val_df)
        test_df = format_df_cls(test_df)

    return train_df, val_df, test_df


def set_model_args(n_classes, num_epoch, in_dim,
                   lr=cfg['transformer']['optimizer']['lr'],
                   train_all_bert=True):
    ## Add arguments:
    model_args = ClassificationArgs(evaluate_during_training=True)
    model_args.num_labels = n_classes
    # model_args.no_cache = True
    # model_args.no_save = True
    model_args.num_train_epochs = num_epoch
    model_args.output_dir = cfg['paths']['result_dir']
    # model_args.cache_dir = cfg['paths']['cache_dir']
    model_args.fp16 = False
    # model_args.fp16_opt_level = 'O1'
    model_args.max_seq_length = cfg['transformer']['max_seq_len']
    # model_args.weight_decay = cfg['transformer']['optimizer']['weight_decay']
    # model_args.learning_rate = cfg['transformer']['optimizer']['lr']
    # model_args.adam_epsilon = cfg['transformer']['optimizer']['adam_epsilon']
    # model_args.warmup_ratio = cfg['transformer']['optimizer']['warmup_ratio']
    # model_args.warmup_steps = cfg['transformer']['optimizer']['warmup_steps']
    # model_args.max_grad_norm = cfg['transformer']['optimizer']['max_grad_norm']
    model_args.train_batch_size = cfg['transformer']['train_batch_size']
    # model_args.gradient_accumulation_steps = cfg['transformer']['gradient_accumulation_steps']
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.overwrite_output_dir = True
    model_args.eval_batch_size = cfg['transformer']['eval_batch_size']
    # model_args.evaluate_during_training = True
    # model_args.evaluate_during_training_verbose = True
    # model_args.evaluate_during_training_silent = False
    model_args.evaluate_each_epoch = True
    model_args.use_early_stopping = True
    model_args.save_model_every_epoch = False
    model_args.save_eval_checkpoints = False
    model_args.save_optimizer_and_scheduler = False
    model_args.reprocess_input_data = True
    # model_args.evaluate_during_training_steps = 3000
    model_args.save_steps = 10000
    model_args.n_gpu = 1
    if lr is not None:
        model_args.learning_rate = lr
    model_args.threshold = 0.5
    model_args.early_stopping_patience = 3
    if not train_all_bert:
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
        logger.warning(f'Training classifier only.')

    ## GCN params:
    model_args.in_dim = in_dim
    model_args.hid_dim = model_args.in_dim
    model_args.out_dim = model_args.in_dim

    logger.warning(f'Model Arguments: {model_args}')

    return model_args


def get_input_file_paths(data_name=cfg['data']['name'], data_dir=dataset_dir):
    source_name, target_name = data_name.split("-")
    logger.info(f"Data name [{data_name}] with source [{source_name}] and"
                f" target [{target_name}]")

    train_name = source_name + "_train"
    val_name = source_name + "_val"
    test_name = target_name + "_test"
    unlabelled_source_name = source_name + "_train"
    if exists(join(data_dir, source_name + "_unlabeled")):
        unlabelled_source_name = source_name + "_unlabeled"
    unlabelled_target_name = target_name + "_train"
    if exists(join(data_dir, target_name + "_unlabeled")):
        unlabelled_target_name = target_name + "_unlabeled"

    logger.info(f"\nTrain [{train_name}] \nVal [{val_name}] \nTest "
                f"[{test_name}] \nSource Unlabelled ["
                f"{unlabelled_source_name}] \nTarget Unlabelled "
                f"[{unlabelled_target_name}]")

    train_path = join(data_dir, train_name)
    val_path = join(data_dir, val_name)
    test_path = join(data_dir, test_name)
    unlabelled_source_path = join(data_dir, unlabelled_source_name)
    unlabelled_target_path = join(data_dir, unlabelled_target_name)

    logger.info(f"\nTrain [{train_path}] \nVal [{val_path}] \nTest "
                f"[{test_path}] \nSource Unlabelled ["
                f"{unlabelled_source_path}] \nTarget Unlabelled "
                f"[{unlabelled_target_path}]")

    return train_name, val_name, test_name, train_path, val_path, test_path,\
           unlabelled_source_name, unlabelled_target_name, unlabelled_source_path, unlabelled_target_path,\
           source_name, target_name


def main(args, multi_label=cfg['data']['multi_label']):
    train_name, val_name, test_name, train_path, val_path, test_path,\
    unlabelled_source_name, unlabelled_target_name, unlabelled_source_path,\
    unlabelled_target_path, source_name, target_name = get_input_file_paths(
        data_dir=dataset_dir)

    train_df, val_df, test_df = read_data(
        train_name=train_name, val_name=val_name, test_name=test_name,
        data_dir=dataset_dir, train_portion=args.train_portion,
        format_input=multi_label)

    if multi_label:
        num_classes = len(train_df.labels.to_list()[0])
    else:
        if type(train_df.labels.to_list()[0]) is list:
            num_classes = len(train_df.labels.to_list()[0])
        else:
            num_classes = 1
    model_args = set_model_args(n_classes=num_classes, lr=args.lr,
                                num_epoch=args.num_train_epochs, in_dim=768,
                                train_all_bert=True)

    model_args.multi_label = multi_label

    clf = ClassificationModel(
        model_type=args.model_type, model_name=args.model_name,
        num_labels=num_classes, args=model_args,
        use_cuda=args.use_cuda, cuda_device=cuda_device)

    if args.use_graph:
        clf = setup_graph_component(args, clf, train_df, train_name, unlabelled_source_name, unlabelled_target_name)

    eval_dicts = {}
    eval_dicts[target_name] = test_df
    extra_names = [source_name + "_test", ]
    for extra_name in extra_names:
        eval_df = read_csv(data_dir=dataset_dir, data_file=extra_name)

        if multi_label:
            eval_df = format_df_cls(eval_df)
        elif type(eval_df.labels.iloc[0]) == str:
            eval_df.labels = eval_df.labels.map(ast.literal_eval)
        eval_dicts[extra_name] = eval_df

    logger.info(clf.train_model(train_df, eval_df=eval_dicts[target_name],
                                multi_label=multi_label, weighted_f1=weighted_f1))

    # extra_names = [args.source_name + "_test", ]
    # for extra_name in extra_names:
    #     extra_df = read_csv(data_dir=dataset_dir, data_file=extra_name)
    #     clf.eval_model(eval_df=extra_df, multi_label=multi_label, weighted_f1=weighted_f1)

    logger.info("Execution Completed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("-g", "--use_graph", default=False, type=bool,
                        help="False is for without graph component.")
    parser.add_argument("-en", "--exp_name", default='bert', type=str)
    parser.add_argument("-c", "--use_cuda", default=True, type=bool)
    parser.add_argument("-m", "--model_name", default=cfg['transformer']['model_name'], type=str)
    parser.add_argument("-mt", "--model_type", default=cfg['transformer']['model_type'], type=str)
    parser.add_argument("-e", "--num_train_epochs", default=cfg['transformer']['num_epoch'], type=int)
    parser.add_argument("-d", "--dataset_name", default=cfg['data']['name'], type=str)
    parser.add_argument("-ml", "--multilingual", default=True, type=bool)
    # parser.add_argument("-s", "--source_name", default=cfg['data']['source_name'], type=str)
    # parser.add_argument("-t", "--target_name", default=cfg['data']['target_name'], type=str)
    parser.add_argument("-p", "--train_portion", default=None, type=float)
    parser.add_argument("-lr", "--lr", default=None, type=float)
    parser.add_argument("-sc", "--seed_count", default=cfg['training']['seed_count'], type=int)

    args = parser.parse_args()
    logger.info(f"Command-line Arguments: {args}")


    def run_main():
        logger.info(f'Run for [{args.seed_count}] SEEDS')
        for _ in range(args.seed_count):
            seed = random.randint(0, 50000)
            logger.info(f'Setting SEED [{seed}]')
            set_all_seeds(seed)
            main(args, multi_label=cfg['data']['multi_label'])


    # exp_name = 'defaultlr_'
    # train_portions = cfg['data']['train_portions']
    # for t_portion in train_portions:
    #     args.train_portion = t_portion
    #     args.exp_name = exp_name + 'train_portion_[' +\
    #                     str(args.train_portion) + ']_'
    #     logger.info(f'Run for TRAIN portion: [{args.train_portion}]')
    #     run_main()

    lrs = cfg['transformer']['lrs']
    train_portions = cfg['data']['train_portions']
    for lr in lrs:
        args.lr = lr
        exp_name = 'lr_[' + str(args.lr) + ']_'
        logger.info(f'Run for lr: [{args.lr}]')
        for t_portion in train_portions:
            args.train_portion = t_portion
            args.exp_name = exp_name + 'train_portion_[' +\
                            str(args.train_portion) + ']_'
            logger.info(f'Run for TRAIN portion: [{args.train_portion}]')
            run_main()

    logger.info(f"Execution completed for {args.seed_count} SEEDs.")
