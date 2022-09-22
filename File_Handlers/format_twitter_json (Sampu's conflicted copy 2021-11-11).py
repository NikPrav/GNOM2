# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Convert twitter output to dataframe.
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

import argparse
import pandas as pd
from os.path import join, exists
from config import configuration as cfg, platform as plat, username as user,\
    dataset_dir
from Logger.logger import logger
from File_Handlers.json_handler import read_json
from multilingual.multilingual_operations import translate_texts


def read_dataset(data_dir=dataset_dir,
                 train_name='anonymyzed_train_data_content.json',
                 val_name='anonymyzed_val_data_content.json',
                 test_name='anonymyzed_test_data_content.json'):
    val_df = pd.read_json(join(data_dir, val_name), lines=True)
    test_df = pd.read_json(join(data_dir, test_name), lines=True)
    train_df = pd.read_json(join(data_dir, train_name), lines=True)
    logger.info(f"Dataset Details:\n "
                f"Train size {train_df.shape}, lang counts: {train_df.lang.value_counts()}"
                f"\n Val size {val_df.shape}, lang counts: {val_df.lang.value_counts()}"
                f"\n Test size {test_df.shape}, lang counts: {test_df.lang.value_counts()}")

    return train_df, val_df, test_df


def translate_examples(df):
    non_eng_df = df[df.lang != 'en']
    non_eng_df_translated = translate_texts(non_eng_df.text.to_list())
    logger.info(non_eng_df_translated)


def twitter_json2csv(df, cols=None, index_col='id'):
    if cols is None:
        cols = ['text']
    ndf = df.filter(cols, axis=1)
    ndf.index = df[index_col]
    return ndf


def get_ids_from_json(path, name):
    """ Reads tweet ids (one per line) from a file.

    @param path:
    @param name:
    @return:
    """
    logger.info("Reading tweet ids from {}", join(path, name))
    # data = json.loads(join(path, name))
    data = read_json(file_path=join(path, name))

    return data["tweet_ids"]



def add_labels(df_labs, df):



def main(args, ):
    train_df, val_df, test_df = read_dataset(
        data_dir=join(args.data_dir, "Multilingual-BERT-Disaster-master/Processed_Data/"),
        train_name='anonymyzed_train_data_content.json',
        val_name='anonymyzed_val_data_content.json',
        test_name='anonymyzed_test_data_content.json')

    # translate_examples(val_df)
    train_df = twitter_json2csv(train_df, cols=['text'], index_col='id')
    val_df = twitter_json2csv(val_df, cols=['text'], index_col='id')
    test_df = twitter_json2csv(test_df, cols=['text'], index_col='id')

    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("-d", "--data_dir", type=str,
                        default=cfg["paths"]['dataset_root'][plat][user])
    parser.add_argument("-f", "--filename", type=str,
                        default="Multilingual-BERT-Disaster-master/"
                                "Processed_Data/anonymyzed_val_data_content.json",
                        help="Takes a txt file with tweet id per line.")
    parser.add_argument("-s", "--secrets_path", type=str, default="secrets.json")
    parser.add_argument("-o", "--output_dir", type=str, default=None)

    args = parser.parse_args()
    # input_filepath = join(args.data_path, args.filename)

    if args.output_dir is None:
        args.output_dir = join(args.data_dir)

    main(args)
