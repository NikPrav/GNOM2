# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Contains system configuration details and other global
variables.

__description__ : Benefit: We can print the configuration on every run and
get the hyper-parameter configuration.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.3 "
__date__        : "04-03-2019"
__copyright__   : "Copyright (c) 2019, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
found in the LICENSE file in the root directory of this source tree."

__classes__     : config

__variables__   : configuration, seed, platform

__methods__     :

__last_modified__:
"""

import json
import subprocess as sp
from os import environ
from os.path import join
from torch import cuda, device

# from Logger.logger import create_logger
# logger = create_logger(logger_name=f"{cfg['data']['name']}")

global seed
seed = 0

global configuration
configuration = {
    "data":         {
        # 'name': 'fire16-smerp17', 'num_classes': 4, 'multi_label': True, # 'source_name': 'fire16', 'target_name': 'smerp17',
        # 'name': 'smerp17-fire16','num_classes': 4, 'multi_label': True, # 'source_name': 'smerp17', 'target_name': 'fire16',

        # 'name': 'mKaggle-mKaggle', 'num_classes': 37, 'multi_label': True, # 'source_name': 'NEQ', 'target_name': 'QFL',
        # 'name': 'TwitterDisaster', 'num_classes': 1, 'multi_label': False,
        'name': '20NG', 'num_classes': 20, 'multi_label': False,
         # 'source_name': 'NEQ', 'target_name': 'QFL',
        # 'name': 'ecuador_en-ecuador_es', 'num_classes': 1, 'multi_label': False, # 'source_name': 'NEQ', 'target_name': 'QFL',

        # 'name': 'NEQ-QFL', 'num_classes': 1, 'multi_label': False, # 'source_name': 'NEQ', 'target_name': 'QFL',
        # 'name': 'QFL-NEQ', 'num_classes': 1, 'multi_label': False, # 'source_name': 'QFL', 'target_name': 'NEQ',

        # 'class_names':    ('0'),
        # 'class_names': ('0', '1', '2', '3'),
        'min_freq':       3,
       'estimate_importance': 'new', 'use_graph': True, 'multilingual': True,
        # 'train_portions': [1.0, 0.5, 0.25, 0.1],
        'train_portions': [0.01],

        'zeroshot':       False,
        'all_test_files': [

            'fire16_test',
            'smerp17_test'
        ],

        
        # 'name':       'TwitterDisaster',
        # 'train':      'TwitterDisaster_train',
        # 'val':        'TwitterDisaster_train',
        # 'test':       'TwitterDisaster_test',
        # "source":     {
        #     'labelled':   'TwitterDisaster_train',
        #     'unlabelled': 'TwitterDisaster_train'
        # },
        # "target":     {
        #     'labelled':   'TwitterDisaster_train',
        #     'unlabelled': 'TwitterDisaster_train'},

        'name':       '20NG',
        'train':      '20NG_train',
        'val':        '20NG_train',
        'test':       '20NG_test',
        "source":     {
            'labelled':   '20NG_train',
            'unlabelled': '20NG_train'
        },
        "target":     {
            'labelled':   '20NG_train',
            'unlabelled': '20NG_train'},    

        'num_classes': 20,
        # 'class_names': ('0', '1', '2', '3'),

    },

    "lstm":  {
        "num_layers":  1,
        "num_linear":  1,
        "bias":        True,
        "batch_first": True,
        "bi":          True,
        "hid_dim":     100,
        "dropout":     0.2,
    },

    "transformer":  {
        "num_epoch":                   2,
        'lrs':                         [1e-5],
        "train_batch_size":            16,
        "eval_batch_size":             128,
        # "model_type":                  "bert",
        # "model_name":                  "bert-base-uncased",
        "model_type":                  "glen_bert",
        "model_name":                  "bert-base-multilingual-cased",
        # "model_name":                  "roberta-base",
        # "model_type":                  "glen_xlmroberta",
        # "model_name":                  "xlm-roberta-base",
        "max_seq_len":                 64,
        'gradient_accumulation_steps': 1,
        "max_vec_len":                 5000,
        "dropout":                     0.1,
        "dropout_external":            0.0,
        "clipnorm":                    2.0,
        "normalize_inputs":            False,
        "kernel_size":                 1,
        "stride":                      1,
        "padding":                     1,
        "context":                     5,
        "classify_count":              0,
        "fce":                         True,
        "optimizer":                   {
            "optimizer_type":          "AdamW",
            "learning_rate_scheduler": "linear_warmup",
            "lr":                      3e-4,
            "lr_decay":                0.,
            "weight_decay":            0.,
            "max_grad_norm":           1.0,
            "adam_epsilon":            1e-8,
            'warmup_ratio':            0.06,
            'warmup_steps':            0.,
            "momentum":                0.9,
            "dampening":               0.9,
            "alpha":                   0.99,
            "rho":                     0.9,
            "centered":                False
        },
        "view_grads":                  False,
        "view_train_precision":        True
    },

    "training":     {
        "seed_count":          1,
        "seed_start":          0,
        "num_epoch":           2,
        "cls_pretrain_epochs": [1, 3],
        "train_batch_size":    16,
        "eval_batch_size":     256,
    },

    "model":        {
        'type':                 'LSTM',
        'lrs':                  [1e-4, 1e-2],
        'mittens_iter':         100,
        "max_sequence_length":  128,
        "dropout":              0.2,
        "dropout_external":     0.0,
        "clipnorm":             1.0,
        "normalize_inputs":     False,
        "kernel_size":          1,
        "stride":               1,
        "padding":              1,
        "context":              10,
        "classify_count":       0,
        "optimizer":            {
            "optimizer_type": "adam",
            "lr":             0.001,
            "lr_decay":       0,
            "weight_decay":   0.0001,
            "momentum":       0.9,
            "dampening":      0.9,
            "alpha":          0.99,
            "rho":            0.9,
            "centered":       False
        },
        "view_grads":           False,
        "view_train_precision": True
    },

    "embeddings":   {
        'embedding_file': 'glove.6B.300d',
        'emb_dim':        300,
    },

    "gnn_params":   {
        "hid_dim":     300,
        "num_heads":   2,
        "padding":     1,
        "stride":      1,
        "kernel_size": 1,
        "bias":        True,
    },

    "prep_vecs":    {
        "max_nb_words":   20000,
        "min_word_count": 1,
        "window":         7,
        "min_freq":       1,
        "negative":       10,
        "num_chunks":     10,
        "idf":            True
    },

    "text_process": {
        "encoding":         'latin-1',
        "sents_chunk_mode": "word_avg",
        "workers":          5
    },
    'pretrain':     {
        'epoch':       80,
        'model_type':  'GCN',
        'save_epochs': [1, 3, 10, 50, 80],
        'min_freq':    3,
        'lr':          8e-4,
        # 'name':        'disaster_binary_pretrain',
        # 'name':        'Amazon_Reviews_Sentiment_books',
        # 'name':        'Amazon_Reviews_Sentiment_dvd',
        # 'name':        'Amazon_Reviews_Sentiment_electronics',
        # 'name':        'Amazon_Reviews_Sentiment_kitchen',
        # 'name':        'Amazon_Reviews_Sentiment_video',

        # 'name':        'pretrain_AF',
        # 'name':        'pretrain_BB',
        # 'name':        'pretrain_KL',
        # 'name':        'pretrain_NE',
        # 'name':        'pretrain_OT',
        # 'name':        'pretrain_QF',
        # 'name':        'pretrain_SH',
        'name':        'pretrain_WE',
        'files':       [

            'AF13_train', 'AF13_val',
            'BB13_train', 'BB13_val',
            'Kaggle_train', 'Kaggle_val',
            'NEQ15_train', 'NEQ15_val',
            'OT13_train', 'OT13_val',
            'QFL13_train', 'QFL13_val',
            'SH12_train', 'SH12_val',
            # 'WTE13_train', 'WTE13_val',

            # 'IEQ12',

            # 'books_train', 'books_val',
            # 'dvd_train', 'dvd_val',
            # 'electronics_train', 'electronics_val',
            # 'kitchen_train', 'kitchen_val',
            # 'video_train', 'video_val',
        ],
    },
    'ecl_pretrain': {
        'epoch':       2,
        'save_epochs': [1, 3, 5, 10, 50],
        'lr':          0.005,
    },

    "paths":        {
        "result_dir":    "results",
        "log_dir":       "logs",
        "cache_dir":     "cache",

        "embedding_dir": {
            "Windows": "D:\\Datasets\\Extreme Classification",
            "OSX":     "/home/cs16resch01001/datasets/Extreme Classification",
            "Linux":   {
                "sam":            "/home/sam/Embeddings",
                "root":           "/home/sam/Embeddings",
                "cs14mtech11017": "/home/cs14mtech11017/Embeddings",
                "cs16resch01001": "/raid/cs16resch01001/Embeddings",
                ## Code path: /home/cs14resch11001/codes/MNXC
                "cs14resch11001": "/raid/ravi/pretrain",
                 "nik":"/home/nik/Documents/sem7/projects/sam/Short-text_GNN/Embeddings"
            }
        },

        'dataset_root':  {
            "Windows": "D:\\Datasets\\Extreme Classification",
            "OSX":     "/home/cs16resch01001/datasets/Extreme Classification",
            "Linux":   {
                "sam":            "/home/sam/Datasets",
                "root":           "/home/sam/Datasets",
                "cs14mtech11017": "/home/cs14mtech11017/Datasets",
                "cs16resch01001": "/raid/cs16resch01001/datasets",
                "cs14resch11001": "/raid/ravi/Datasets/Extreme Classification",
                "nik":"/home/nik/Documents/sem7/projects/sam/Short-text_GNN/Datasets"
            }
        }
    },

    'cuda':         {
        "use_cuda":     {
            "Windows": False,
            "OSX":     False,
            "Linux":   {
                "sam":            False,
                "root":           False,
                "cs14mtech11017": True,
                "cs16resch01001": True,
                "cs14resch11001": True,
                "nik": True
            },
        },
        "cuda_devices": {
            "Windows": False,
            "OSX":     False,
            "Linux":   {
                "sam":            False,
                "root":           False,
                "cs14mtech11017": 0,
                "cs16resch01001": 6,
                "cs14resch11001": 7,
                "nik": 0
            },
        },
    },

    "transformer":  {
        "num_epoch":                   2,
        'lrs':                         [1e-3],
        "train_batch_size":            8,
        "eval_batch_size":             64,
        "model_type":                  "glen_bert",
        "model_name":                  "bert-base-uncased",
        # "model_type":                  "glen_xlmroberta",
        # "model_name":                  "xlm-roberta-base",
        "max_seq_len":                 64,
        'gradient_accumulation_steps': 1,
        "max_vec_len":                 5000,
        "dropout":                     0.1,
        "dropout_external":            0.0,
        "clipnorm":                    2.0,
        "normalize_inputs":            False,
        "kernel_size":                 1,
        "stride":                      1,
        "padding":                     1,
        "context":                     5,
        "classify_count":              0,
        "fce":                         True,
        "optimizer":                   {
            "optimizer_type":          "AdamW",
            "learning_rate_scheduler": "linear_warmup",
            "lr":                      3e-4,
            "lr_decay":                0.,
            "weight_decay":            0.,
            "max_grad_norm":           1.0,
            "adam_epsilon":            1e-8,
            'warmup_ratio':            0.06,
            'warmup_steps':            0.,
            "momentum":                0.9,
            "dampening":               0.9,
            "alpha":                   0.99,
            "rho":                     0.9,
            "centered":                False
        },
        "view_grads":                  False,
        "view_train_precision":        True
    },

    "model":        {
        'type':                 'LSTM',
        'lrs':                  [1e-4, 1e-2],
        'mittens_iter':         100,
        "max_sequence_length":  128,
        "dropout":              0.2,
        "dropout_external":     0.0,
        "clipnorm":             1.0,
        "normalize_inputs":     False,
        "kernel_size":          1,
        "stride":               1,
        "padding":              1,
        "context":              10,
        "classify_count":       0,
        "optimizer":            {
            "optimizer_type": "adam",
            "lr":             0.001,
            "lr_decay":       0,
            "weight_decay":   0.0001,
            "momentum":       0.9,
            "dampening":      0.9,
            "alpha":          0.99,
            "rho":            0.9,
            "centered":       False
        },
        "view_grads":           False,
        "view_train_precision": True
    },

    "embeddings":   {
        'embedding_file': 'glove.6B.300d',
        'emb_dim':        300,
    },

    "lstm_params":  {
        "num_layers":  2,
        "bias":        True,
        "batch_first": True,
        "bi":          True,
        "hid_size":    64,
    },

    "gnn_params":   {
        "hid_dim":     300,
        "num_heads":   2,
        "padding":     1,
        "stride":      1,
        "kernel_size": 1,
        "bias":        True,
    },

    "training":     {
        "seed_count":          1,
        "seed_start":          0,
        "num_epoch":           2,
        "cls_pretrain_epochs": [1, 3],
        "train_batch_size":    16,
        "eval_batch_size":     256,
    },

    "prep_vecs":    {
        "max_nb_words":   20000,
        "min_word_count": 1,
        "window":         7,
        "min_freq":       1,
        "negative":       10,
        "num_chunks":     10,
        "idf":            True
    },

    "text_process": {
        "encoding":         'latin-1',
        "sents_chunk_mode": "word_avg",
        "workers":          5
    },
}


class Config(object):
    """ Contains all configuration details of the project. """

    def __init__(self):
        super(Config, self).__init__()

        self.configuration = configuration

    def get_config(self):
        """

        :return:
        """
        return self.configuration

    def print_config(self, indent=4, sort=True):
        """ Prints the config. """
        print("[{}] : {}".format("Configuration",
                                 json.dumps(self.configuration,
                                            indent=indent,
                                            sort_keys=sort)))

    @staticmethod
    def get_platform():
        """ Returns dataset path based on OS.

        :return: str
        """
        import platform

        if platform.system() == 'Windows':
            return platform.system()
        elif platform.system() == 'Linux':
            return platform.system()
        else:  ## OS X returns name 'Darwin'
            return "OSX"

    @staticmethod
    def get_username():
        """
        :returns the current username.

        :return: string
        """
        try:
            import os, pwd

            username = pwd.getpwuid(os.getuid()).pw_name
        except Exception as e:
            import getpass

            username = getpass.getuser()
        # finally:
        #     username = os.environ.get('USER')

        return username


config_cls = Config()
# config_cls.print_config()

global platform
platform = config_cls.get_platform()
global username
username = config_cls.get_username()
global dataset_dir
dataset_dir = join(configuration["paths"]['dataset_root'][platform][username],
                   configuration['data']['name'])

global pretrain_dir
pretrain_dir = join(configuration['paths']['dataset_root'][platform][username],
                    configuration['pretrain']['name'])

global emb_dir
emb_dir = configuration['paths']['embedding_dir'][platform][username]


# def get_gpu_details():
#     if cuda.is_available() and configuration['cuda']["use_cuda"][platform][username]:
#         _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
#
#         COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
#         memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
#         memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
#         device_id = memory_free_values.index(max(memory_free_values))
#
#         print(f'Allocated: {round(cuda.memory_allocated(0) / 1024 ** 3, 1)}GB')
#         print(f'Cached: {round(cuda.memory_reserved(0) / 1024 ** 3, 1)}GB')
#         print(f'Selected GPU: [{device_id}] as available RAM: {memory_free_values}')
#         return device_id
#
#
# def set_cuda_device(device_id=None):
#     if cuda.is_available() and configuration['cuda']["use_cuda"][platform][username]:
#         if device_id is None:
#             device_id = get_gpu_details()
#         environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
#         cuda.set_device(device_id)
#         cuda_device = device(f'cuda:' + str(device_id) if cuda.is_available() else 'cpu')
#         print(f'set_device: {device_id} == device: {cuda_device} == current_device:'
#               f' {cuda.current_device()} out of device_count: {cuda.device_count()}')
#     else:
#         print('CUDA NOT SUPPORTED')
#         cuda_device = 'cpu'
#         device_id = -1
#
#     return device_id, cuda_device
#
#
# global cuda_device
# global device_id
# cuda_device = 'cpu'
# device_id = None
# # # if cuda.is_available() and configuration['cuda']["use_cuda"][platform][username]:
# # # # device_id, cuda_device = set_cuda_device()
# # # cuda_device = device(f'cuda:' + str(device_id) if cuda.is_available() else 'cpu')
# #
# if cuda.is_available() and configuration['cuda']["use_cuda"][platform][username]:
# #     device_id = get_gpu_details()
#     cuda_device = device(f'cuda:' + str(device_id) if cuda.is_available() and configuration['cuda']["use_cuda"][
#     platform][username] else 'cpu')

def get_gpu_details():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    device_id = memory_free_values.index(max(memory_free_values))
    print(f'Selected GPU: [{device_id}] as available RAM: {memory_free_values}')
    return device_id


global cuda_device
global device_id
# device = device(f'cuda:' + str(configuration['cuda']['cuda_devices'][platform][
#                                    username]) if
#                 cuda.is_available() else 'cpu')
device_id = get_gpu_details()
cuda_device = device(f'cuda:' + str(device_id) if cuda.is_available() else 'cpu')


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
