# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : GLEN_BERT
__description__ : Replaced GAT with BERT in GLEN
__project__     : Tweet_GNN_inductive
__classes__     : Tweet_GNN_inductive
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "19/09/21"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

from __future__ import absolute_import, division, print_function

import ast
import math
import random
import warnings
from dataclasses import asdict
# from multiprocessing import cpu_count
import tempfile
from typing import *
from json import dumps
from copy import deepcopy
from collections import Counter
from os import environ, makedirs, listdir
from os.path import join, exists
from pathlib import Path
from torch import cuda, save, load, nn, no_grad, tensor, long, set_num_threads,\
    manual_seed, device, Tensor, qint8, quantization, mean, cat, stack, mm,\
    triu, add, argmax, softmax as t_softmax, from_numpy
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.functional import normalize

import numpy as np
import pandas as pd
from scipy.stats import mode, pearsonr
from scipy.special import softmax
from sklearn.metrics import (
    confusion_matrix,
    label_ranking_average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
    roc_curve,
    auc,
    average_precision_score,
)
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from tqdm.contrib import tenumerate
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.optimization import AdamW, Adafactor
from transformers import (
    AlbertConfig,
    AlbertTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertTokenizerFast,
    BertweetTokenizer,
    # BigBirdConfig,
    # BigBirdTokenizer,
    # BigBirdForSequenceClassification,
    CamembertConfig,
    CamembertTokenizerFast,
    DebertaConfig,
    DebertaForSequenceClassification,
    DebertaTokenizer,
    DistilBertConfig,
    DistilBertTokenizerFast,
    # ElectraConfig,
    # ElectraTokenizerFast,
    FlaubertConfig,
    FlaubertTokenizer,
    LayoutLMConfig,
    LayoutLMTokenizerFast,
    LongformerConfig,
    LongformerTokenizerFast,
    MPNetConfig,
    MPNetForSequenceClassification,
    MPNetTokenizerFast,
    MobileBertConfig,
    MobileBertTokenizerFast,
    RobertaConfig,
    RobertaTokenizerFast,
    SqueezeBertConfig,
    SqueezeBertForSequenceClassification,
    SqueezeBertTokenizerFast,
    WEIGHTS_NAME,
    XLMConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizerFast,
    XLMTokenizer,
    XLNetConfig,
    XLNetTokenizerFast,
)
from transformers.convert_graph_to_onnx import convert, quantize

from stf_classification.classification_utils import (
    InputExample,
    LazyClassificationDataset,
    ClassificationDataset,
    convert_examples_to_features,
    # load_hf_dataset,
)
from stf_classification.transformer_models.albert_model import (
    AlbertForSequenceClassification,
)
from stf_classification.transformer_models.bert_model import (
    BertForSequenceClassification,
)
from stf_classification.transformer_models.camembert_model import (
    CamembertForSequenceClassification,
)
from stf_classification.transformer_models.distilbert_model import (
    DistilBertForSequenceClassification,
)
from stf_classification.transformer_models.flaubert_model import (
    FlaubertForSequenceClassification,
)
from stf_classification.transformer_models.layoutlm_model import (
    LayoutLMForSequenceClassification,
)
from stf_classification.transformer_models.longformer_model import (
    LongformerForSequenceClassification,
)
from stf_classification.transformer_models.mobilebert_model import (
    MobileBertForSequenceClassification,
)
from stf_classification.transformer_models.roberta_model import (
    RobertaForSequenceClassification,
)
from stf_classification.transformer_models.xlm_model import (
    XLMForSequenceClassification,
)
from stf_classification.transformer_models.xlm_roberta_model import (
    XLMRobertaForSequenceClassification,
)
from stf_classification.transformer_models.xlnet_model import (
    XLNetForSequenceClassification,
)
from stf_config.global_args import global_args
from stf_config.model_args import ClassificationArgs
from stf_config.utils import sweep_config_to_sweep_values
# from simpletransformers.custom_models.models import ElectraForSequenceClassification

# from File_Handlers.csv_handler import read_csv, read_csvs
# from File_Handlers.pkl_handler import save_pickle, load_pickle
# from File_Handlers.json_handler import save_json, read_json, read_labelled_json
# from Text_Encoder.finetune_static_embeddings import glove2dict, calculate_cooccurrence_mat,\
#     train_mittens, preprocess_and_find_oov
# from Data_Handlers.token_handler_nx import Token_Dataset_nx
# from Label_Propagation_PyTorch.label_propagation import fetch_all_nodes, label_propagation
from config import configuration as cfg, platform as plat, username as user,\
    dataset_dir, pretrain_dir, cuda_device
# from Text_Processesor.build_corpus_vocab import get_dataset_fields
from Graph_Construction.create_subword_token_graph import construct_token_graph
from Layers.glen_bert_classifier import GLEN_BERT_Classifier, BertForGLEN
from Layers.glen_xlmroberta_classifier import XLMRobertaForGLEN
from Layers.gcn_classifiers import GCN
from Layers.bilstm_classifiers import BiLSTM_Classifier, Importance_Estimator
from Metrics.metrics import weighted_f1
from File_Handlers.csv_handler import read_csv, read_csvs
from Utils.utils import set_all_seeds, count_parameters, dot
from Logger.logger import logger

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

environ["TOKENIZERS_PARALLELISM"] = "false"

MODELS_WITHOUT_CLASS_WEIGHTS_SUPPORT = ["squeezebert", "deberta", "mpnet"]

MODELS_WITH_EXTRA_SEP_TOKEN = [
    "roberta",
    "camembert",
    "xlmroberta", "glen_xlmroberta",
    "longformer",
    "mpnet",
]

MODELS_WITH_ADD_PREFIX_SPACE = [
    "roberta",
    "camembert",
    "xlmroberta", "glen_xlmroberta",
    "longformer",
    "mpnet",
]

MODELS_WITHOUT_SLIDING_WINDOW_SUPPORT = ["squeezebert"]


class ClassificationModel:
    def __init__(self, model_type=cfg['transformer']['model_type'],
                 model_name=cfg['transformer']['model_name'],
                 use_graph=cfg['data']['use_graph'], estimate_importance=cfg['data']['estimate_importance'],
                 tokenizer_type=None, tokenizer_name=None, num_labels=cfg['data']['num_classes'],
                 weight=None, args=None, use_cuda=True, cuda_device=-1,
                 onnx_execution_provider=None, exp_name="GLEN_BERT", **kwargs):
        """
        Initializes a ClassificationModel model.

        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers 
            compatible pre-trained model, a community model, or the path to a directory containing model files.
            tokenizer_type: The type of tokenizer (auto, bert, xlnet, xlm, roberta, distilbert, etc.) to use. If a 
            string is passed, Simple Transformers will try to initialize a tokenizer class from the available 
            MODEL_CLASSES.
                                Alternatively, a Tokenizer class (subclassed from PreTrainedTokenizer) can be passed.
            tokenizer_name: The name/path to the tokenizer. If the tokenizer_type is not specified, the model_type 
            will be used to determine the type of the tokenizer.
            num_labels (optional): The number of labels or classes in the dataset.
            weight (optional): A list of length num_labels containing the weights to assign to each label for loss 
            calculation.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a 
            dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            onnx_execution_provider (optional): ExecutionProvider to use with ONNX Runtime. Will use CUDA (if 
            use_cuda) or CPU (if use_cuda is False) by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options 
            specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "albert":          (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
            "auto":            (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer),
            "bert":            (BertConfig, BertForSequenceClassification, BertTokenizerFast),
            "glen_bert":       (BertConfig, BertForGLEN, BertTokenizerFast),
            "bertweet":        (
                RobertaConfig,
                RobertaForSequenceClassification,
                BertweetTokenizer,
            ),
            # "bigbird": (
            #     BigBirdConfig,
            #     BigBirdForSequenceClassification,
            #     BigBirdTokenizer,
            # ),
            "camembert":       (
                CamembertConfig,
                CamembertForSequenceClassification,
                CamembertTokenizerFast,
            ),
            "deberta":         (
                DebertaConfig,
                DebertaForSequenceClassification,
                DebertaTokenizer,
            ),
            "distilbert":      (
                DistilBertConfig,
                DistilBertForSequenceClassification,
                DistilBertTokenizerFast,
            ),
            # "electra": (
            #     ElectraConfig,
            #     ElectraForSequenceClassification,
            #     ElectraTokenizerFast,
            # ),
            "flaubert":        (
                FlaubertConfig,
                FlaubertForSequenceClassification,
                FlaubertTokenizer,
            ),
            "layoutlm":        (
                LayoutLMConfig,
                LayoutLMForSequenceClassification,
                LayoutLMTokenizerFast,
            ),
            "longformer":      (
                LongformerConfig,
                LongformerForSequenceClassification,
                LongformerTokenizerFast,
            ),
            "mobilebert":      (
                MobileBertConfig,
                MobileBertForSequenceClassification,
                MobileBertTokenizerFast,
            ),
            "mpnet":           (MPNetConfig, MPNetForSequenceClassification, MPNetTokenizerFast),
            "roberta":         (
                RobertaConfig,
                RobertaForSequenceClassification,
                RobertaTokenizerFast,
            ),
            "squeezebert":     (
                SqueezeBertConfig,
                SqueezeBertForSequenceClassification,
                SqueezeBertTokenizerFast,
            ),
            "xlm":             (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
            "xlmroberta":      (
                XLMRobertaConfig,
                XLMRobertaForSequenceClassification,
                XLMRobertaTokenizerFast,
            ),
            "glen_xlmroberta": (
                XLMRobertaConfig,
                XLMRobertaForGLEN,
                XLMRobertaTokenizerFast,
            ),
            "xlnet":           (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizerFast),
        }

        self.combine = 'concat'
        self.use_graph = use_graph
        self.estimate_importance = estimate_importance
        self.model_type = model_type
        self.model_name = model_name
        self.multi_label = args.multi_label
        self.class_weights = None

        args.num_labels = num_labels

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ClassificationArgs):
            self.args = args

        self.exp_name = 'Model: [' + model_name + '], ' +\
                        'graph: [' + str(self.use_graph) + '], ' +\
                        'importance: [' + str(self.estimate_importance) + '], ' +\
                        exp_name

        logger.fatal(f'Experiment: {self.exp_name}')

        if (
                model_type in MODELS_WITHOUT_SLIDING_WINDOW_SUPPORT
                and self.args.sliding_window
        ):
            raise ValueError(
                "{} does not currently support sliding window".format(model_type)
            )

        if self.args.thread_count:
            set_num_threads(self.args.thread_count)

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                cuda.manual_seed_all(self.args.manual_seed)

        if self.args.labels_list:
            if num_labels:
                assert num_labels == len(self.args.labels_list)
            if self.args.labels_map:
                try:
                    assert list(self.args.labels_map.keys()) == self.args.labels_list
                except AssertionError:
                    assert [
                               int(key) for key in list(self.args.labels_map.keys())
                           ] == self.args.labels_list
                    self.args.labels_map = {
                        int(key): value for key, value in self.args.labels_map.items()
                    }
            else:
                self.args.labels_map = {
                    label: i for i, label in enumerate(self.args.labels_list)
                }
        else:
            len_labels_list = 2 if not num_labels else num_labels
            self.args.labels_list = [i for i in range(len_labels_list)]

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

        if tokenizer_type is not None:
            if isinstance(tokenizer_type, str):
                _, _, tokenizer_class = MODEL_CLASSES[tokenizer_type]
            else:
                tokenizer_class = tokenizer_type

        if num_labels:
            self.config = config_class.from_pretrained(
                model_name, num_labels=num_labels, **self.args.config
            )
            self.num_labels = num_labels
        else:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
            if self.args.num_labels:
                self.num_labels = self.args.num_labels
            else:
                self.num_labels = self.config.num_labels

        if model_type in MODELS_WITHOUT_CLASS_WEIGHTS_SUPPORT and weight is not None:
            raise ValueError(
                "{} does not currently support class weights".format(model_type)
            )
        else:
            self.weight = weight

        if use_cuda:
            if cuda.is_available():
                if cuda_device == -1:
                    self.device = device("cuda")
                else:
                    self.device = device(f"{cuda_device}")
            else:
                logger.warning(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
                self.device = "cpu"
        else:
            self.device = "cpu"

        logger.info(f"Cuda device: {cuda_device}")

        if self.args.onnx:
            from onnxruntime import InferenceSession, SessionOptions

            if not onnx_execution_provider:
                onnx_execution_provider = (
                    "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"
                )

            options = SessionOptions()

            if self.args.dynamic_quantize:
                model_path = quantize(Path(join(model_name, "onnx_model.onnx")))
                self.text_representer = InferenceSession(
                    model_path.as_posix(), options, providers=[onnx_execution_provider]
                )
            else:
                model_path = join(model_name, "onnx_model.onnx")
                self.text_representer = InferenceSession(
                    model_path, options, providers=[onnx_execution_provider]
                )
        else:
            if not self.args.quantized_model:
                if self.weight:
                    self.text_representer = model_class.from_pretrained(
                        model_name,
                        config=self.config,
                        weight=Tensor(self.weight).to(self.device),
                        **kwargs,
                    )
                else:
                    self.text_representer = model_class.from_pretrained(
                        model_name, config=self.config, **kwargs
                    )
            else:
                quantized_weights = load(
                    join(model_name, "pytorch_model.bin")
                )
                if self.weight:
                    self.text_representer = model_class.from_pretrained(
                        None,
                        config=self.config,
                        state_dict=quantized_weights,
                        weight=Tensor(self.weight).to(self.device),
                    )
                else:
                    self.text_representer = model_class.from_pretrained(
                        None, config=self.config, state_dict=quantized_weights
                    )

            if self.args.dynamic_quantize:
                self.text_representer = quantization.quantize_dynamic(
                    self.text_representer, {nn.Linear}, dtype=qint8
                )
            if self.args.quantized_model:
                self.text_representer.load_state_dict(quantized_weights)
            if self.args.dynamic_quantize:
                self.args.quantized_model = True

        self.results = {}

        if not use_cuda:
            self.args.fp16 = False

        if self.args.fp16:
            try:
                from torch.cuda import amp
            except AttributeError:
                raise AttributeError(
                    "fp16 requires Pytorch >= 1.6. Please update Pytorch or turn off fp16."
                )

        if tokenizer_name is None:
            tokenizer_name = model_name

        if tokenizer_name in [
            "vinai/bertweet-base",
            "vinai/bertweet-covid19-base-cased",
            "vinai/bertweet-covid19-base-uncased",
        ]:
            self.tokenizer = tokenizer_class.from_pretrained(
                tokenizer_name,
                do_lower_case=self.args.do_lower_case,
                normalization=True,
                **kwargs,
            )
        else:
            self.tokenizer = tokenizer_class.from_pretrained(
                tokenizer_name, do_lower_case=self.args.do_lower_case, **kwargs
            )

        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(
                self.args.special_tokens_list, special_tokens=True
            )
            self.text_representer.resize_token_embeddings(len(self.tokenizer))

        self.args.model_name = model_name
        self.args.model_type = model_type
        self.args.tokenizer_name = tokenizer_name
        self.args.tokenizer_type = tokenizer_type

        if model_type in ["camembert", "xlmroberta", "glen_xlmroberta"]:
            warnings.warn(
                f"use_multiprocessing automatically disabled as {model_type}"
                " fails when using multiprocessing for feature conversion."
            )
            self.args.use_multiprocessing = False

        if self.args.wandb_project and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.args.wandb_project = None

        final_dim = self.args.out_dim
        if self.use_graph:
            ## Graph Layers:
            self.graph_featurizer = GCN(in_dim=self.args.in_dim, hid_dim=self.args.hid_dim,
                                        out_dim=self.args.out_dim)
            if self.combine == 'concat':
                final_dim = 2 * self.args.out_dim
            elif self.combine == 'avg':
                final_dim = self.args.out_dim
            else:
                raise NotImplementedError(f'combine supports either concat or avg.'
                                          f' [{self.combine}] provided.')

            # self.global_readout = mean()

            if self.estimate_importance == 'none':
                pass
            elif self.estimate_importance == 'old':
                self.graph_importance_estimator = Importance_Estimator(final_dim, 1)
            elif self.estimate_importance == 'new':
                self.estimator = nn.MultiheadAttention(embed_dim=self.args.out_dim, num_heads=3)
        self.bilstm_classifier = BiLSTM_Classifier(final_dim, num_labels, hid_dim=cfg['lstm']['hid_dim'],
                                                   n_layers=cfg['lstm']['num_layers'],
                                                   dropout=cfg['lstm']['dropout'], num_linear=cfg['lstm']['num_linear'])

    def forward(self, inputs):
        # self.bert2gcn_id_map = get_bert2gcn_token_map(self.tokenizer.vocab,
        #                                               self.joint_vocab['str2idx_map'])

        token_type_ids = None
        if 'token_type_ids' in inputs:
            token_type_ids = inputs['token_type_ids']

        outputs = self.text_representer(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=token_type_ids)

        text_outputs = outputs.last_hidden_state
        text_pooled = outputs.pooler_output
        # if self.model_type == 'glen_xlmroberta':
        #     text_output = text_outputs.last_hidden_state
        #     text_pooled = text_outputs.pooler_output
        # else:
        #     text_output = text_outputs

        # text_pooled = text_outputs.pooler_output
        if self.use_graph:
            X = self.graph_featurizer(self.A, self.X)

            gcn_ids = get_bert2gcn_id_mapped_examples(
                inputs['input_ids'], self.tokenizer.bert_vocab_i2s,
                self.joint_vocab['str2idx_map'],
                special_ids=self.tokenizer.all_special_ids)

            graph_outputs = X[gcn_ids]

            ## Estimate global emp importance using neural network:
            if self.estimate_importance == 'none':
                pass
            elif self.estimate_importance == 'old':
                ## Use readout over global embeddings:
                graph_pooled = mean(graph_outputs, dim=1)
                # graph_pooled = self.global_readout(graph_outputs)
                alpha = self.graph_importance_estimator(text_pooled, graph_pooled)
                graph_outputs = alpha.unsqueeze(1) * graph_outputs
            else:
                ## get estimator mask; reformulate input mask.
                ## False = attends, True values are omitted in Pytorch implimentation
                estimator_mask = (inputs['attention_mask'] - 1).bool()

                ## convert node vectors to seq_len first; Pytorch >= 1.10 supports batch_first.
                graph_outputs_seq_first = graph_outputs.permute(1, 0, 2)

                _, graph_outputs_weights = self.estimator(
                    text_pooled.unsqueeze(0), graph_outputs_seq_first,
                    graph_outputs_seq_first, key_padding_mask=estimator_mask)

                graph_outputs = graph_outputs_weights.permute(0, 2, 1) * graph_outputs
            # else:
            #     raise NotImplementedError(f'Use either "none", "old" or "new" estimator,'
            #                               f' [{self.estimate_importance}] provided.')

            if self.combine == 'concat':
                combined_emb = cat([graph_outputs, text_outputs], dim=2)
            elif self.combine == 'avg':
                combined_emb = mean(stack([graph_outputs, text_outputs]), dim=0)
            else:
                raise NotImplementedError(f'"combine" supports either [concat] or [avg].'
                                          f' [{self.combine}] provided.')
            text_outputs = combined_emb

        logits = self.bilstm_classifier(text_outputs)

        labels = inputs['labels']

        if labels is not None:
            labels = labels.to(self.device).float()

            if self.class_weights is not None:
                weight = self.class_weights.to(labels.device)
            else:
                weight = None

            ## Multi-Label Classification
            if self.args.multi_label and self.num_labels > 1:
                loss_fct = BCEWithLogitsLoss(pos_weight=weight)
                loss = loss_fct(
                    # logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
                    logits, labels
                )
            ## Binary Classification
            elif self.num_labels == 1:
                loss_fct = BCEWithLogitsLoss(pos_weight=weight)
                loss = loss_fct(logits.view(-1), labels.view(-1))
            ## Multi-Class Classification
            else:
                loss_fct = CrossEntropyLoss(weight=weight)
                # if labels.shape[-1] > 1 and len(inputs["labels"].shape) > 1:
                    # ce_labels = argmax(labels, dim=1).long()
                # loss = loss_fct(logits.view(-1, self.num_labels), ce_labels.view(-1))
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())
            # text_outputs = (loss,) + text_outputs

            return loss, logits, text_outputs
        return None, logits, text_outputs

    def train_model(self, train_df, multi_label=cfg['data']['multi_label'], class_weights=None,
                    output_dir=None, show_running_loss=True, args=None,
                    eval_dfs=None, verbose=True, **kwargs):
        """
        Trains the model using 'train_df' and evaluates on each 'eval_dfs' if provided.

        Args:
            train_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, 
            it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, 
            and the second column containing the label. The model will be trained on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. 
            Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the 
            model.
            eval_df (optional): List of eval_df: A DataFrame against which evaluation will be performed when 
            evaluate_during_training 
            is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of 
            metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, 
                        and the second parameter will be the predictions.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress 
            scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        if class_weights is not None:
            self.class_weights = class_weights

        if args:
            self.args.update_from_dict(args)

        if self.args.silent:
            show_running_loss = False

        if self.args.evaluate_during_training and eval_dfs is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_df is not specified."
                " Pass eval_df to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if (
                exists(output_dir)
                and listdir(output_dir)
                and not self.args.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set overwrite_output_dir: True to automatically overwrite.".format(
                    output_dir
                )
            )
        self._move_model_to_device()

        if self.args.use_hf_datasets:
            if self.args.sliding_window:
                raise ValueError(
                    "HuggingFace Datasets cannot be used with sliding window."
                )
            if self.args.model_type == "layoutlm":
                raise NotImplementedError(
                    "HuggingFace Datasets support is not implemented for LayoutLM models"
                )
            train_dataset = load_hf_dataset(
                train_df, self.tokenizer, self.args, multi_label=self.args.multi_label
            )
        elif isinstance(train_df, str) and self.args.lazy_loading:
            if self.args.sliding_window:
                raise ValueError("Lazy loading cannot be used with sliding window.")
            if self.args.model_type == "layoutlm":
                raise NotImplementedError(
                    "Lazy loading is not implemented for LayoutLM models"
                )
            train_dataset = LazyClassificationDataset(
                train_df, self.tokenizer, self.args
            )
        else:
            if self.args.lazy_loading:
                raise ValueError(
                    "Input must be given as a path to a file when using lazy loading"
                )
            # if "trues" in train_df.columns:
            #     train_examples = (
            #         train_df["text"].astype(str).tolist(),
            #         (
            #             train_df["labels"].tolist(),
            #             train_df["trues"].tolist(),
            #             train_df["logit0"].tolist(),
            #             train_df["logit1"].tolist()
            #         ),
            #         train_df.index.astype(str),
            #     )
            elif "text" in train_df.columns and "labels" in train_df.columns:
                if self.args.model_type == "layoutlm":
                    train_examples = [
                        InputExample(text, None, label, x0, y0, x1, y1, idx)
                        for i, (text, label, x0, y0, x1, y1, idx) in enumerate(
                            zip(
                                train_df["text"].astype(str),
                                train_df["labels"],
                                train_df["x0"],
                                train_df["y0"],
                                train_df["x1"],
                                train_df["y1"],
                                train_df.index.astype(str),
                            )
                        )
                    ]
                else:
                    train_examples = (
                        train_df["text"].astype(str).tolist(),
                        train_df["labels"].tolist(),
                        train_df.index.astype(str),
                    )
            elif "text_a" in train_df.columns and "text_b" in train_df.columns:
                if self.args.model_type == "layoutlm":
                    raise ValueError("LayoutLM cannot be used with sentence-pair tasks")
                else:
                    train_examples = (
                        train_df["text_a"].astype(str).tolist(),
                        train_df["text_b"].astype(str).tolist(),
                        train_df["labels"].tolist(),
                        train_df.index.astype(str),
                    )
            else:
                warnings.warn(
                    "Dataframe headers not specified. Falling back to using column 0 as text and column 1 as labels."
                )
                train_examples = (
                    train_df.index.astype(str),
                    train_df.iloc[:, 0].astype(str).tolist(),
                    train_df.iloc[:, 1].tolist(),
                )
            train_dataset = self.load_and_cache_examples(
                train_examples, verbose=verbose
            )
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.train(
            train_dataloader,
            output_dir,
            multi_label=multi_label,
            show_running_loss=show_running_loss,
            eval_dfs=eval_dfs,
            verbose=verbose,
            **kwargs,
        )

        # model_to_save = self.text_representer.module if hasattr(self.text_representer, "module") else
        # self.text_representer
        # model_to_save.save_pretrained(output_dir)
        # self.tokenizer.save_pretrained(output_dir)
        # save(self.args, join(output_dir, "training_args.bin"))
        self.save_model(model=self.text_representer)

        if verbose:
            logger.info(
                " Training of {} model complete. Saved to {}.".format(
                    self.args.model_type, output_dir
                )
            )

        return global_step, training_details

    def train(self, train_dataloader, output_dir, multi_label=cfg['data']['multi_label'],
              show_running_loss=True, eval_df=None, verbose=True,
              eval_dfs=None, **kwargs):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """
        model = self.text_representer
        args = self.args

        total_param_count = 0

        total_param_count += count_parameters(model)

        tb_writer = SummaryWriter(logdir=args.tensorboard_dir)

        t_total = (
                len(train_dataloader)
                // args.gradient_accumulation_steps
                * args.num_train_epochs
        )

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []

        ## Add representer params to optimizer:
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [
                p for n, p in model.named_parameters() if n in params
            ]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params":       [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                               and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params":       [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                               and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        ## Add LSTM params:
        lstm_group = {}
        # lstm_group["lr"] = 0.001
        lstm_group["params"] = [p for n, p in self.bilstm_classifier.named_parameters()]
        optimizer_grouped_parameters.append(lstm_group)
        total_param_count += count_parameters(self.bilstm_classifier)

        if self.use_graph:
            ## Add GCN params:
            gcn_group = {}
            # gcn_group["lr"] = 0.001
            gcn_group["params"] = [p for n, p in self.graph_featurizer.named_parameters()]
            optimizer_grouped_parameters.append(gcn_group)
            total_param_count += count_parameters(self.graph_featurizer)

            if self.estimate_importance == 'none':
                pass
            elif self.estimate_importance == 'old':
                ## Add Global Importance Estimator params:
                gie_group = {}
                # gie_group["lr"] = 0.001
                gie_group["params"] = [p for n, p in self.graph_importance_estimator.named_parameters()]
                optimizer_grouped_parameters.append(gie_group)
                total_param_count += count_parameters(self.graph_importance_estimator)

            else:
                ## Add Global Importance Estimator params:
                gie_group = {}
                # gie_group["lr"] = 0.001
                gie_group["params"] = [p for n, p in self.estimator.named_parameters()]
                optimizer_grouped_parameters.append(gie_group)
                total_param_count += count_parameters(self.estimator)

        logger.info(f'Number of parameters: [{total_param_count}]')

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = (
            warmup_steps if args.warmup_steps == 0 else args.warmup_steps
        )

        if args.optimizer == "AdamW":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon,
            )
        elif args.optimizer == "Adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adafactor_eps,
                clip_threshold=args.adafactor_clip_threshold,
                decay_rate=args.adafactor_decay_rate,
                beta1=args.adafactor_beta1,
                weight_decay=args.weight_decay,
                scale_parameter=args.adafactor_scale_parameter,
                relative_step=args.adafactor_relative_step,
                warmup_init=args.adafactor_warmup_init,
            )
            logger.info("Using Adafactor for T5")
        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                    args.optimizer
                )
            )

        if args.scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)

        elif args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps
            )

        elif args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
            )

        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "polynomial_decay_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                lr_end=args.polynomial_decay_schedule_lr_end,
                power=args.polynomial_decay_schedule_power,
            )

        else:
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        if args.n_gpu > 1:
            model = nn.DataParallel(model)
            self.bilstm_classifier = nn.DataParallel(self.bilstm_classifier)
            # self.global_readout = nn.DataParallel(self.global_readout)
            if self.use_graph:
                self.graph_featurizer = nn.DataParallel(self.graph_featurizer)

                if self.estimate_importance == 'none':
                    pass
                elif self.estimate_importance == 'old':
                    self.graph_importance_estimator = nn.DataParallel(
                        self.graph_importance_estimator)
                elif self.estimate_importance == 'new':
                    self.estimator = nn.DataParallel(self.estimator)

        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0
        )
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0
        current_loss = "Initializing"

        if args.model_name and exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (
                        len(train_dataloader) // args.gradient_accumulation_steps
                )
                steps_trained_in_current_epoch = global_step % (
                        len(train_dataloader) // args.gradient_accumulation_steps
                )

                logger.info(
                    "   Continuing training from checkpoint, will skip to saved global_step"
                )
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info(
                    "   Will skip the first %d steps in the current epoch",
                    steps_trained_in_current_epoch,
                )
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(
                multi_label, **kwargs
            )

        if args.wandb_project:
            if not wandb.setup().settings.sweep_id:
                logger.info(" Initializing WandB run for training.")
                wandb.init(
                    project=args.wandb_project,
                    config={**asdict(args), "repo": "simpletransformers"},
                    **args.wandb_kwargs,
                )
            wandb.watch(self.text_representer)

        if self.args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for _ in train_iterator:
            model.train()
            self.bilstm_classifier.train()

            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(
                f"Epoch {epoch_number + 1} of {args.num_train_epochs}"
            )
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                inputs = self._get_inputs_dict(batch)

                if epoch_number == cfg["transformer"]["num_epoch"] - 2:
                    print(epoch_number)

                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        loss = outputs[0]
                else:
                    outputs = self.forward(inputs)
                    # if self.use_graph:
                    #     outputs = self.forward(inputs)
                    # else:
                    #     outputs = model(**inputs)
                    # outputs = model(**inputs, multi_label=True)
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    loss = outputs[0]

                if args.n_gpu > 1:
                    loss = (loss.mean()
                            )  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if self.args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        scaler.unscale_(optimizer)
                    if args.optimizer == "AdamW":
                        nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                    if self.args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        tb_writer.add_scalar(
                            "lr", scheduler.get_last_lr()[0], global_step
                        )
                        tb_writer.add_scalar(
                            "loss",
                            (tr_loss - logging_loss) / args.logging_steps,
                            global_step,
                        )
                        logging_loss = tr_loss
                        if args.wandb_project or self.is_sweeping:
                            wandb.log(
                                {
                                    "Training loss": current_loss,
                                    "lr":            scheduler.get_last_lr()[0],
                                    "global_step":   global_step,
                                }
                            )

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )

                        self.save_model(
                            output_dir_current, optimizer, scheduler, model=model
                        )

                    if args.evaluate_during_training and (
                            args.evaluate_during_training_steps > 0
                            and global_step % args.evaluate_during_training_steps == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ = self.eval_model(
                            eval_dfs,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent,
                            wandb_log=False,
                            # epoch=epoch_number,
                            **kwargs,
                        )

                        logger.info(f"Result at epoch {epoch_number}:"
                                    f"\n{dumps(results, indent=4)}")

                        results = list(results.items())[0][1]

                        # for key, value in results.items():
                        #     tb_writer.add_scalar(
                        #         "eval_{}".format(key), value, global_step
                        #     )

                        output_dir_current = join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )

                        if args.save_eval_checkpoints:
                            self.save_model(
                                output_dir_current,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            try:
                                training_progress_scores[key].append(results[key])
                            except KeyError as e:
                                logger.error(e)
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            join(
                                args.output_dir, "training_progress_scores.csv"
                            ),
                            index=False,
                        )

                        if args.wandb_project or self.is_sweeping:
                            wandb.log(self._get_last_metrics(training_progress_scores))

                        if not best_eval_metric:
                            best_eval_metric = results[args.early_stopping_metric]
                            self.save_model(
                                args.best_model_dir,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )
                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if (
                                    best_eval_metric - results[args.early_stopping_metric]
                                    > args.early_stopping_delta
                            ):
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if (
                                            early_stopping_counter
                                            < args.early_stopping_patience
                                    ):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(
                                                f" No improvement in {args.early_stopping_metric}"
                                            )
                                            logger.info(
                                                f" Current step: {early_stopping_counter}"
                                            )
                                            logger.info(
                                                f" Early stopping patience: {args.early_stopping_patience}"
                                            )
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args.early_stopping_patience} steps reached"
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        else:
                            if (
                                    results[args.early_stopping_metric] - best_eval_metric
                                    > args.early_stopping_delta
                            ):
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if (
                                            early_stopping_counter
                                            < args.early_stopping_patience
                                    ):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(
                                                f" No improvement in {args.early_stopping_metric}"
                                            )
                                            logger.info(
                                                f" Current step: {early_stopping_counter}"
                                            )
                                            logger.info(
                                                f" Early stopping patience: {args.early_stopping_patience}"
                                            )
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args.early_stopping_patience} steps reached"
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        model.train()

            epoch_number += 1
            output_dir_current = join(
                output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number)
            )

            if args.save_model_every_epoch or args.evaluate_during_training:
                makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:
                all_results, _, _ = self.eval_model(
                    eval_dfs,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    wandb_log=False,
                    multi_label=multi_label,
                    **kwargs,
                )

                self.results[epoch_number] = all_results
                results = list(all_results.items())[0][1]

                self.save_model(
                    output_dir_current, optimizer, scheduler, results=results
                )

                for name, result in all_results.items():
                    logger.info(
                        f"Result epoch: [{epoch_number}], {self.exp_name}, "
                        f"Test: [{name}], W-F1: [{result['weighted_f1']}]:"
                        f"\n{dumps(result, indent=4)}")

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(
                    join(args.output_dir, "training_progress_scores.csv"),
                    index=False,
                )

                if args.wandb_project or self.is_sweeping:
                    wandb.log(self._get_last_metrics(training_progress_scores))

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    self.save_model(
                        args.best_model_dir,
                        optimizer,
                        scheduler,
                        model=model,
                        results=results,
                    )
                if best_eval_metric and args.early_stopping_metric_minimize:
                    if (
                            best_eval_metric - results[args.early_stopping_metric]
                            > args.early_stopping_delta
                    ):
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )
                        early_stopping_counter = 0
                    else:
                        if (
                                args.use_early_stopping
                                and args.early_stopping_consider_epochs
                        ):
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(
                                        f" No improvement in {args.early_stopping_metric}"
                                    )
                                    logger.info(
                                        f" Current step: {early_stopping_counter}"
                                    )
                                    logger.info(
                                        f" Early stopping patience: {args.early_stopping_patience}"
                                    )
                            else:
                                if verbose:
                                    logger.info(
                                        f" Patience of {args.early_stopping_patience} steps reached"
                                    )
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )
                else:
                    if (
                            results[args.early_stopping_metric] - best_eval_metric
                            > args.early_stopping_delta
                    ):
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )
                        early_stopping_counter = 0
                    else:
                        if (
                                args.use_early_stopping
                                and args.early_stopping_consider_epochs
                        ):
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(
                                        f" No improvement in {args.early_stopping_metric}"
                                    )
                                    logger.info(
                                        f" Current step: {early_stopping_counter}"
                                    )
                                    logger.info(
                                        f" Early stopping patience: {args.early_stopping_patience}"
                                    )
                            else:
                                if verbose:
                                    logger.info(
                                        f" Patience of {args.early_stopping_patience} steps reached"
                                    )
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )

        return (
            global_step,
            tr_loss / global_step
            if not self.args.evaluate_during_training
            else training_progress_scores,
        )

    def eval_model(self, eval_dicts=None, eval_df=None, multi_label=cfg['data']['multi_label'],
                   output_dir=None, verbose=True, silent=False, wandb_log=True,
                   **kwargs):
        """
        Evaluates the model on eval_df. Saves results to output_dir.

        Args:
            eval_dicts: Dict of {name : eval_df} - eval_df: Pandas Dataframe containing at least two columns. If the 
            Dataframe has a header, 
            it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, 
            and the second column containing the label. The model will be evaluated on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            wandb_log: If True, evaluation results will be logged to wandb.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of 
            metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, 
                        and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results.
            model_outputs: List of model outputs for each row in eval_df
            wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model
        """  # noqa: ignore flake8"

        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        results = OrderedDict()
        model_outputs = []
        wrong_preds = []
        if eval_dicts:
            for eval_name, eval_df in eval_dicts.items():
                result, model_output, wrong_pred = self.evaluate(
                    eval_df, output_dir, multi_label=multi_label,
                    verbose=verbose, silent=silent, wandb_log=wandb_log, **kwargs)

                # logger.info(f"Result for Data [{eval_name}] at epoch [{epoch}]:"
                #             f"\n{dumps(result, indent=4)}")
                results[eval_name] = result
                model_outputs.append(model_output)
                wrong_preds.append(wrong_pred)
        elif eval_df is not None:
            result, model_outputs, wrong_preds = self.evaluate(
                eval_df, output_dir, multi_label=multi_label,
                verbose=verbose, silent=silent, wandb_log=wandb_log, **kwargs)
            results['result'] = result

            # logger.info(f"Result for at epoch {epoch}:"
            #             f"\n{dumps(results, indent=4)}")
        else:
            raise ValueError("Either eval_dicts or eval_df should be provided.")

        # self.results.update(results)
        # self.results[epoch] = results

        # if verbose:
        #     logger.info(self.results)

        return results, model_outputs, wrong_preds

    def evaluate(self, eval_df, output_dir, verbose=True, silent=False,
                 wandb_log=True, **kwargs):
        """
        Evaluates the model on eval_df.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """
        multi_label = self.args.multi_label
        model = self.text_representer
        args = self.args
        eval_output_dir = output_dir

        results = {}
        if self.args.use_hf_datasets:
            if self.args.sliding_window:
                raise ValueError(
                    "HuggingFace Datasets cannot be used with sliding window."
                )
            if self.args.model_type == "layoutlm":
                raise NotImplementedError(
                    "HuggingFace Datasets support is not implemented for LayoutLM models"
                )
            eval_dataset = load_hf_dataset(
                eval_df, self.tokenizer, self.args, multi_label=multi_label
            )
            eval_examples = None
        elif isinstance(eval_df, str) and self.args.lazy_loading:
            if self.args.model_type == "layoutlm":
                raise NotImplementedError(
                    "Lazy loading is not implemented for LayoutLM models"
                )
            eval_dataset = LazyClassificationDataset(eval_df, self.tokenizer, self.args)
            eval_examples = None
        else:
            if self.args.lazy_loading:
                raise ValueError(
                    "Input must be given as a path to a file when using lazy loading"
                )

            if "text" in eval_df.columns and "labels" in eval_df.columns:
                if self.args.model_type == "layoutlm":
                    eval_examples = [
                        InputExample(text, None, label, x0, y0, x1, y1, idx)
                        for i, (text, label, x0, y0, x1, y1, idx) in enumerate(
                            zip(
                                eval_df["text"].astype(str),
                                eval_df["labels"],
                                eval_df["x0"],
                                eval_df["y0"],
                                eval_df["x1"],
                                eval_df["y1"],
                                eval_df.index.astype(str),
                            )
                        )
                    ]
                else:
                    eval_examples = (
                        eval_df["text"].astype(str).tolist(),
                        eval_df["labels"].tolist(),
                        eval_df.index.astype(str),
                    )
            elif "text_a" in eval_df.columns and "text_b" in eval_df.columns:
                if self.args.model_type == "layoutlm":
                    raise ValueError("LayoutLM cannot be used with sentence-pair tasks")
                else:
                    eval_examples = (
                        eval_df["text_a"].astype(str).tolist(),
                        eval_df["text_b"].astype(str).tolist(),
                        eval_df["labels"].tolist(),
                        eval_df.index.astype(str),
                    )
            else:
                warnings.warn(
                    "Dataframe headers not specified. Falling back to using column 0 as text and column 1 as labels."
                )
                eval_examples = (
                    eval_df.iloc[:, 0].astype(str).tolist(),
                    eval_df.iloc[:, 1].tolist(),
                    eval_df.index.astype(str),
                )

            if args.sliding_window:
                eval_dataset, window_counts = self.load_and_cache_examples(
                    eval_examples, evaluate=True, verbose=verbose, silent=silent
                )
            else:
                eval_dataset = self.load_and_cache_examples(
                    eval_examples, evaluate=True, verbose=verbose, silent=silent
                )
        makedirs(eval_output_dir, exist_ok=True)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        if args.n_gpu > 1:
            model = nn.DataParallel(model)
            self.bilstm_classifier = nn.DataParallel(self.bilstm_classifier)
            # self.global_readout = nn.DataParallel(self.global_readout)
            if self.use_graph:
                self.graph_featurizer = nn.DataParallel(self.graph_featurizer)

                if self.estimate_importance == 'none':
                    pass
                elif self.estimate_importance == 'old':
                    self.graph_importance_estimator = nn.DataParallel(
                        self.graph_importance_estimator)
                elif self.estimate_importance == 'new':
                    self.estimator = nn.DataParallel(self.estimator)

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(eval_dataloader)
        preds = np.empty((len(eval_dataset), self.num_labels))
        if multi_label:
            out_label_ids = np.empty((len(eval_dataset), self.num_labels))
        else:
            out_label_ids = np.empty((len(eval_dataset)))
        model.eval()
        self.bilstm_classifier.eval()
        # self.global_readout = nn.DataParallel(self.global_readout)
        if self.use_graph:
            self.graph_featurizer.eval()

            if self.estimate_importance == 'none':
                pass
            elif self.estimate_importance == 'old':
                self.graph_importance_estimator.eval()
            elif self.estimate_importance == 'new':
                self.estimator.eval()

        if self.args.fp16:
            from torch.cuda import amp

        for i, batch in enumerate(
                tqdm(eval_dataloader, disable=args.silent or silent,
                     desc="Running Evaluation")):
            # batch = tuple(t.to(device) for t in batch)

            with no_grad():
                inputs = self._get_inputs_dict(batch)

                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        tmp_eval_loss, logits = outputs[:2]
                else:
                    outputs = self.forward(inputs)
                    # if self.use_graph:
                    #     outputs = self.forward(inputs)
                    # else:
                    #     outputs = model(**inputs)
                    # outputs = self.forward(inputs)
                    tmp_eval_loss, logits = outputs[:2]

                if multi_label:
                    logits = logits.sigmoid()
                elif inputs["labels"].shape[-1] > 1 and len(inputs["labels"].shape) > 1:
                    logits = t_softmax(logits, dim=-1)
                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()
            
                logits = t_softmax(logits, dim=-1)

            nb_eval_steps += 1

            start_index = self.args.eval_batch_size * i
            end_index = (
                start_index + self.args.eval_batch_size
                if i != (n_batches - 1)
                else len(eval_dataset)
            )
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            ## Only take class index in case of multi-class:
            if inputs["labels"].shape[-1] > 1 and len(inputs["labels"].shape) > 1 and not multi_label:
                out_label_ids[start_index:end_index] = (
                    argmax(inputs["labels"].detach(), dim=-1).cpu().numpy()
                )
            else:
                out_label_ids[start_index:end_index] = (
                    inputs["labels"].detach().cpu().numpy()
                )

            # if preds is None:
            #     preds = logits.detach().cpu().numpy()
            #     out_label_ids = inputs["labels"].detach().cpu().numpy()
            # else:
            #     preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            #     out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        if args.sliding_window:
            count = 0
            window_ranges = []
            for n_windows in window_counts:
                window_ranges.append([count, count + n_windows])
                count += n_windows

            preds = [
                preds[window_range[0]: window_range[1]]
                for window_range in window_ranges
            ]
            out_label_ids = [
                out_label_ids[i]
                for i in range(len(out_label_ids))
                if i in [window[0] for window in window_ranges]
            ]

            model_outputs = preds

            preds = [np.argmax(pred, axis=1) for pred in preds]
            final_preds = []
            for pred_row in preds:
                val_freqs_desc = Counter(pred_row).most_common()
                if (
                        len(val_freqs_desc) > 1
                        and val_freqs_desc[0][1] == val_freqs_desc[1][1]
                ):
                    final_preds.append(args.tie_value)
                else:
                    final_preds.append(val_freqs_desc[0][0])
            preds = np.array(final_preds)
        elif not multi_label and args.regression is True:
            preds = np.squeeze(preds)
            model_outputs = preds
        else:
            model_outputs = preds

            if inputs["labels"].shape[-1] > 1 and len(inputs["labels"].shape) > 1 and not multi_label:
                preds = np.argmax(preds, axis=1)

        result, wrong = self.compute_metrics(
            preds, model_outputs, out_label_ids, eval_examples, **kwargs)
        result["eval_loss"] = eval_loss
        # result["eval_name"] = data_name
        # result["epoch"] = epoch
        results.update(result)

        # logger.info(f"Result for Data {result['eval_name']} at epoch "
        #             f"{result['epoch']}:\n{dumps(result, indent=4)}")

        # output_eval_file = join(eval_output_dir, "eval_results.txt")
        # with open(output_eval_file, "w") as writer:
        #     for key in sorted(result.keys()):
        #         writer.write("{} = {}\n".format(key, str(result[key])))

        if (
                self.args.wandb_project
                and wandb_log
                and not multi_label
                and not self.args.regression
        ):
            if not wandb.setup().settings.sweep_id:
                logger.info(" Initializing WandB run for evaluation.")
                wandb.init(
                    project=args.wandb_project,
                    config={**asdict(args), "repo": "simpletransformers"},
                    **args.wandb_kwargs,
                )
            if not args.labels_map:
                self.args.labels_map = {i: i for i in range(self.num_labels)}

            labels_list = sorted(list(self.args.labels_map.keys()))
            inverse_labels_map = {
                value: key for key, value in self.args.labels_map.items()
            }

            truth = [inverse_labels_map[out] for out in out_label_ids]

            # Confusion Matrix
            wandb.sklearn.plot_confusion_matrix(
                truth, [inverse_labels_map[pred] for pred in preds], labels=labels_list,
            )

            if not self.args.sliding_window:
                # ROC`
                wandb.log({"roc": wandb.plots.ROC(truth, model_outputs, labels_list)})

                # Precision Recall
                wandb.log(
                    {
                        "pr": wandb.plots.precision_recall(
                            truth, model_outputs, labels_list
                        )
                    }
                )

        return results, model_outputs, wrong

    def load_and_cache_examples(
            self, examples, evaluate=False, no_cache=False, multi_label=cfg['data']['multi_label'],
            verbose=True, silent=False):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        process_count = self.args.process_count

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not multi_label and args.regression:
            output_mode = "regression"
        else:
            output_mode = "classification"

        if not no_cache:
            makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"
        if args.sliding_window or self.args.model_type == "layoutlm":
            cached_features_file = join(
                args.cache_dir,
                "cached_{}_{}_{}_{}_{}".format(
                    mode,
                    args.model_type,
                    args.max_seq_length,
                    self.num_labels,
                    len(examples),
                ),
            )

            if exists(cached_features_file) and (
                    (not args.reprocess_input_data and not no_cache)
                    or (mode == "dev" and args.use_cached_eval_features and not no_cache)
            ):
                features = load(cached_features_file)
                if verbose:
                    logger.info(
                        f" Features loaded from cache at {cached_features_file}"
                    )
            else:
                if verbose:
                    logger.info(" Converting to features started. Cache is not used.")
                    if args.sliding_window:
                        logger.info(" Sliding window enabled")

                if self.args.model_type != "layoutlm":
                    if len(examples) == 3:
                        examples = [
                            InputExample(text_a, text_b, label, idx)
                            for i, (text_a, text_b, label, idx) in enumerate(zip(*examples))
                        ]
                    else:
                        examples = [
                            InputExample(text_a, None, label, idx)
                            for i, (text_a, label, idx) in enumerate(zip(*examples))
                        ]

                # If labels_map is defined, then labels need to be replaced with ints
                if self.args.labels_map and not self.args.regression:
                    for example in examples:
                        if multi_label:
                            example.label = [
                                self.args.labels_map[label] for label in example.label
                            ]
                        else:
                            example.label = self.args.labels_map[example.label]

                features = convert_examples_to_features(
                    examples,
                    args.max_seq_length,
                    tokenizer,
                    output_mode,
                    # XLNet has a CLS token at the end
                    cls_token_at_end=bool(args.model_type in ["xlnet"]),
                    cls_token=tokenizer.cls_token,
                    cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                    sep_token=tokenizer.sep_token,
                    # RoBERTa uses an extra separator b/w pairs of sentences,
                    # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                    sep_token_extra=args.model_type in MODELS_WITH_EXTRA_SEP_TOKEN,
                    # PAD on the left for XLNet
                    pad_on_left=bool(args.model_type in ["xlnet"]),
                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                    pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                    process_count=process_count,
                    multi_label=multi_label,
                    silent=args.silent or silent,
                    use_multiprocessing=args.use_multiprocessing_for_evaluation,
                    sliding_window=args.sliding_window,
                    flatten=not evaluate,
                    stride=args.stride,
                    add_prefix_space=args.model_type in MODELS_WITH_ADD_PREFIX_SPACE,
                    # avoid padding in case of single example/online inferencing to decrease execution time
                    pad_to_max_length=bool(len(examples) > 1),
                    args=args,
                )
                if verbose and args.sliding_window:
                    logger.info(
                        f" {len(features)} features created from {len(examples)} samples."
                    )

                if not no_cache:
                    save(features, cached_features_file)

            if args.sliding_window and evaluate:
                features = [
                    [feature_set] if not isinstance(feature_set, list) else feature_set
                    for feature_set in features
                ]
                window_counts = [len(sample) for sample in features]
                features = [
                    feature for feature_set in features for feature in feature_set
                ]

            all_input_ids = tensor(
                [f.input_ids for f in features], dtype=long
            )
            all_input_mask = tensor(
                [f.input_mask for f in features], dtype=long
            )
            all_segment_ids = tensor(
                [f.segment_ids for f in features], dtype=long
            )

            if self.args.model_type == "layoutlm":
                all_bboxes = tensor(
                    [f.bboxes for f in features], dtype=long
                )

            if output_mode == "classification":
                all_label_ids = tensor(
                    [f.label_id for f in features], dtype=long
                )
            elif output_mode == "regression":
                all_label_ids = tensor(
                    [f.label_id for f in features], dtype=float
                )

            if self.args.model_type == "layoutlm":
                dataset = TensorDataset(
                    all_input_ids,
                    all_input_mask,
                    all_segment_ids,
                    all_label_ids,
                    all_bboxes,
                )
            else:
                dataset = TensorDataset(
                    all_input_ids, all_input_mask, all_segment_ids, all_label_ids
                )

            if args.sliding_window and evaluate:
                return dataset, window_counts
            else:
                return dataset
        else:
            dataset = ClassificationDataset(
                examples,
                self.tokenizer,
                self.args,
                mode=mode,
                multi_label=multi_label,
                output_mode=output_mode,
                no_cache=no_cache,
            )
            return dataset

    def compute_metrics(self, preds, model_outputs, labels, eval_examples=None,
                        multi_label=cfg['data']['multi_label'], **kwargs):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            preds: Model predictions
            model_outputs: Model outputs
            labels: Ground truth labels
            eval_examples: List of examples on which evaluation was performed
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of 
            metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, 
                        and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results.
            For non-binary classification, the dictionary format is: (Matthews correlation coefficient, tp, tn, fp, fn).
            For binary classification, the dictionary format is: (Matthews correlation coefficient, tp, tn, fp, fn, 
            AUROC, AUPRC).
            wrong: List of InputExample objects corresponding to each incorrect prediction by the model
        """  # noqa: ignore flake8"
        multi_label = self.args.multi_label
        
        preds = preds.argmax(1)

        assert len(preds) == len(labels)

        extra_metrics = {}
        for metric, func in kwargs.items():
            if self.num_labels > 1 and not multi_label:
                ## Setting threshold None for multi-class:
                threshold = None
            else:
                threshold = 0.5
            try:
                extra_metrics[metric] = func(labels, preds, threshold=threshold)
            except TypeError as e:
                logger.error(e)

        if multi_label:
            threshold_values = self.args.threshold if self.args.threshold else 0.5
            if isinstance(threshold_values, list):
                mismatched = labels != [
                    [
                        self._threshold(pred, threshold_values[i])
                        for i, pred in enumerate(example)
                    ]
                    for example in preds
                ]
            else:
                mismatched = labels != [
                    [self._threshold(pred, threshold_values) for pred in example]
                    for example in preds
                ]
        else:
            mismatched = labels != preds

        if eval_examples:
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        if multi_label:
            label_ranking_score = label_ranking_average_precision_score(labels, preds)
            return {**{"LRAP": label_ranking_score}, **extra_metrics}, wrong
        elif self.args.regression:
            return {**extra_metrics}, wrong

        # mcc = 0.0
        mcc = matthews_corrcoef(labels, preds)
        if self.text_representer.num_labels == 2:
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            if self.args.sliding_window:
                return (
                    {
                        **{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn},
                        **extra_metrics,
                    },
                    wrong,
                )
            else:
                scores = np.array([softmax(element)[1] for element in model_outputs])
                fpr, tpr, thresholds = roc_curve(labels, scores)
                auroc = auc(fpr, tpr)
                auprc = average_precision_score(labels, scores)
                return (
                    {
                        **{
                            "mcc":   mcc,
                            "tp":    tp,
                            "tn":    tn,
                            "fp":    fp,
                            "fn":    fn,
                            "auroc": auroc,
                            "auprc": auprc,
                        },
                        **extra_metrics,
                    },
                    wrong,
                )
        else:
            return {**{"mcc": mcc}, **extra_metrics}, wrong

    def predict(self, to_predict, multi_label=cfg['data']['multi_label']):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.

        Returns:
            preds: A python list of the predictions (0 or 1) for each text.
            model_outputs: A python list of the raw model outputs for each text.
        """

        model = self.text_representer
        args = self.args
        multi_label = args.multi_label

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = np.empty((len(to_predict), self.num_labels))
        if multi_label:
            out_label_ids = np.empty((len(to_predict), self.num_labels))
        else:
            out_label_ids = np.empty((len(to_predict)))

        if not multi_label and self.args.onnx:
            model_inputs = self.tokenizer.batch_encode_plus(
                to_predict, return_tensors="pt", padding=True, truncation=True
            )

            for i, (input_ids, attention_mask) in enumerate(
                    zip(model_inputs["input_ids"], model_inputs["attention_mask"])
            ):
                input_ids = input_ids.unsqueeze(0).detach().cpu().numpy()
                attention_mask = attention_mask.unsqueeze(0).detach().cpu().numpy()
                inputs_onnx = {"input_ids": input_ids, "attention_mask": attention_mask}

                # Run the model (None = get all the outputs)
                output = self.text_representer.run(None, inputs_onnx)

                preds[i] = output[0]
                # if preds is None:
                #     preds = output[0]
                # else:
                #     preds = np.append(preds, output[0], axis=0)

            model_outputs = preds
            preds = np.argmax(preds, axis=1)

        else:
            self._move_model_to_device()
            dummy_label = (
                0
                if not self.args.labels_map
                else next(iter(self.args.labels_map.keys()))
            )

            if multi_label:
                dummy_label = [dummy_label for i in range(self.num_labels)]

            if args.n_gpu > 1:
                model = nn.DataParallel(model)
                self.bilstm_classifier = nn.DataParallel(self.bilstm_classifier)
                # self.global_readout = nn.DataParallel(self.global_readout)
                if self.use_graph:
                    self.graph_featurizer = nn.DataParallel(self.graph_featurizer)

                    if self.estimate_importance == 'none':
                        pass
                    elif self.estimate_importance == 'old':
                        self.graph_importance_estimator = nn.DataParallel(
                            self.graph_importance_estimator)
                    elif self.estimate_importance == 'new':
                        self.estimator = nn.DataParallel(self.estimator)

            if isinstance(to_predict[0], list):
                eval_examples = (
                    *zip(*to_predict),
                    [dummy_label for i in range(len(to_predict))],
                )
            else:
                eval_examples = (
                    to_predict,
                    [dummy_label for i in range(len(to_predict))],
                )

            if args.sliding_window:
                eval_dataset, window_counts = self.load_and_cache_examples(
                    eval_examples, evaluate=True, no_cache=True
                )
                preds = np.empty((len(eval_dataset), self.num_labels))
                if multi_label:
                    out_label_ids = np.empty((len(eval_dataset), self.num_labels))
                else:
                    out_label_ids = np.empty((len(eval_dataset)))
            else:
                eval_dataset = self.load_and_cache_examples(
                    eval_examples, evaluate=True, multi_label=multi_label, no_cache=True
                )

            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(
                eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
            )

            if self.args.fp16:
                from torch.cuda import amp

            self.bilstm_classifier.eval()
            # self.global_readout = nn.DataParallel(self.global_readout)
            if self.use_graph:
                self.graph_featurizer.eval()

                if self.estimate_importance == 'none':
                    pass
                elif self.estimate_importance == 'old':
                    self.graph_importance_estimator.eval()
                elif self.estimate_importance == 'new':
                    self.estimator.eval()

            if self.config.output_hidden_states:
                # model.eval()
                preds = None
                out_label_ids = None
                for i, batch in enumerate(
                        tqdm(
                            eval_dataloader, disable=args.silent, desc="Running Prediction"
                        )
                ):
                    # batch = tuple(t.to(self.device) for t in batch)
                    with no_grad():
                        inputs = self._get_inputs_dict(batch, no_hf=True)

                        if self.args.fp16:
                            with amp.autocast():
                                outputs = model(**inputs)
                                tmp_eval_loss, logits = outputs[:2]
                        else:
                            outputs = self.forward(inputs)
                            # if self.use_graph:
                            #     outputs = self.forward(inputs)
                            # else:
                            #     outputs = model(**inputs)
                            tmp_eval_loss, logits = outputs[:2]
                        embedding_outputs, layer_hidden_states = (
                            outputs[2][0],
                            outputs[2][1:],
                        )

                        if multi_label:
                            logits = logits.sigmoid()

                        if self.args.n_gpu > 1:
                            tmp_eval_loss = tmp_eval_loss.mean()
                        eval_loss += tmp_eval_loss.item()

                    nb_eval_steps += 1

                    if preds is None:
                        preds = logits.detach().cpu().numpy()
                        out_label_ids = inputs["labels"].detach().cpu().numpy()
                        all_layer_hidden_states = np.array(
                            [
                                state.detach().cpu().numpy()
                                for state in layer_hidden_states
                            ]
                        )
                        all_embedding_outputs = embedding_outputs.detach().cpu().numpy()
                    else:
                        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                        out_label_ids = np.append(
                            out_label_ids,
                            inputs["labels"].detach().cpu().numpy(),
                            axis=0,
                        )
                        all_layer_hidden_states = np.append(
                            all_layer_hidden_states,
                            np.array(
                                [
                                    state.detach().cpu().numpy()
                                    for state in layer_hidden_states
                                ]
                            ),
                            axis=1,
                        )
                        all_embedding_outputs = np.append(
                            all_embedding_outputs,
                            embedding_outputs.detach().cpu().numpy(),
                            axis=0,
                        )
            else:
                n_batches = len(eval_dataloader)
                for i, batch in enumerate(tqdm(eval_dataloader, disable=args.silent)):
                    # model.eval()
                    # batch = tuple(t.to(device) for t in batch)

                    with no_grad():
                        inputs = self._get_inputs_dict(batch, no_hf=True)

                        if self.args.fp16:
                            with amp.autocast():
                                outputs = model(**inputs)
                                tmp_eval_loss, logits = outputs[:2]
                        else:
                            outputs = self.forward(inputs)
                            # if self.use_graph:
                            #     outputs = self.forward(inputs)
                            # else:
                            #     outputs = model(**inputs)
                            tmp_eval_loss, logits = outputs[:2]

                        if multi_label:
                            logits = logits.sigmoid()

                        if self.args.n_gpu > 1:
                            tmp_eval_loss = tmp_eval_loss.mean()
                        eval_loss += tmp_eval_loss.item()

                    nb_eval_steps += 1

                    start_index = self.args.eval_batch_size * i
                    end_index = (
                        start_index + self.args.eval_batch_size
                        if i != (n_batches - 1)
                        else len(eval_dataset)
                    )
                    preds[start_index:end_index] = logits.detach().cpu().numpy()
                    out_label_ids[start_index:end_index] = (
                        inputs["labels"].detach().cpu().numpy()
                    )

                    # if preds is None:
                    #     preds = logits.detach().cpu().numpy()
                    #     out_label_ids = inputs["labels"].detach().cpu().numpy()
                    # else:
                    #     preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    #     out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps

            if args.sliding_window:
                count = 0
                window_ranges = []
                for n_windows in window_counts:
                    window_ranges.append([count, count + n_windows])
                    count += n_windows

                preds = [
                    preds[window_range[0]: window_range[1]]
                    for window_range in window_ranges
                ]

                model_outputs = preds

                preds = [np.argmax(pred, axis=1) for pred in preds]
                final_preds = []
                for pred_row in preds:
                    mode_pred, counts = mode(pred_row)
                    if len(counts) > 1 and counts[0] == counts[1]:
                        final_preds.append(args.tie_value)
                    else:
                        final_preds.append(mode_pred[0])
                preds = np.array(final_preds)
            elif not multi_label and args.regression is True:
                preds = np.squeeze(preds)
                model_outputs = preds
            else:
                model_outputs = preds
                if multi_label:
                    if isinstance(args.threshold, list):
                        threshold_values = args.threshold
                        preds = [
                            [
                                self._threshold(pred, threshold_values[i])
                                for i, pred in enumerate(example)
                            ]
                            for example in preds
                        ]
                    else:
                        preds = [
                            [self._threshold(pred, args.threshold) for pred in example]
                            for example in preds
                        ]
                else:
                    preds = np.argmax(preds, axis=1)

        if self.args.labels_map and not self.args.regression:
            inverse_labels_map = {
                value: key for key, value in self.args.labels_map.items()
            }
            preds = [inverse_labels_map[pred] for pred in preds]

        if self.config.output_hidden_states:
            return preds, model_outputs, all_embedding_outputs, all_layer_hidden_states
        else:
            return preds, model_outputs

    def convert_to_onnx(self, output_dir=None, set_onnx_arg=True):
        """Convert the model to ONNX format and save to output_dir

        Args:
            output_dir (str, optional): If specified, ONNX model will be saved to output_dir (else args.output_dir 
            will be used). Defaults to None.
            set_onnx_arg (bool, optional): Updates the model args to set onnx=True. Defaults to True.
        """  # noqa
        if not output_dir:
            output_dir = join(self.args.output_dir, "onnx")
        makedirs(output_dir, exist_ok=True)

        if listdir(output_dir):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Output directory for onnx conversion must be empty.".format(
                    output_dir
                )
            )

        onnx_model_name = join(output_dir, "onnx_model.onnx")

        with tempfile.TemporaryDirectory() as temp_dir:
            self.save_model(output_dir=temp_dir, model=self.text_representer)

            convert(
                framework="pt",
                model=temp_dir,
                tokenizer=self.tokenizer,
                output=Path(onnx_model_name),
                pipeline_name="sentiment-analysis",
                opset=11,
            )

        self.args.onnx = True
        self.tokenizer.save_pretrained(output_dir)
        self.config.save_pretrained(output_dir)
        self.save_model_args(output_dir)

    def _threshold(self, x, threshold):
        if x >= threshold:
            return 1
        return 0

    def _move_model_to_device(self):
        self.text_representer.to(self.device)
        self.bilstm_classifier.to(self.device)
        # self.global_readout.to(self.device)

        if self.use_graph:
            self.graph_featurizer.to(self.device)
            if self.estimate_importance == 'none':
                pass
            elif self.estimate_importance == 'old':
                self.graph_importance_estimator.to(self.device)
            elif self.estimate_importance == 'new':
                self.estimator.to(self.device)

    def _get_inputs_dict(self, batch, no_hf=False):
        if self.args.use_hf_datasets and not no_hf:
            return {key: value.to(self.device) for key, value in batch.items()}
        if isinstance(batch[0], dict):
            inputs = {
                key: value.squeeze(1).to(self.device) for key, value in batch[0].items()
            }
            inputs["labels"] = batch[1].to(self.device)
        else:
            batch = tuple(t.to(self.device) for t in batch)

            inputs = {
                "input_ids":      batch[0],
                "attention_mask": batch[1],
                "labels":         batch[3],
            }

            # XLM, DistilBERT and RoBERTa don't use segment_ids
            if self.args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2]
                    if self.args.model_type in ["bert", "xlnet", "albert", "layoutlm"]
                    else None
                )

        if self.args.model_type == "layoutlm":
            inputs["bbox"] = batch[4]

        return inputs

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _create_training_progress_scores(self, multi_label, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        if multi_label:
            training_progress_scores = {
                "global_step": [],
                "LRAP":        [],
                "train_loss":  [],
                "eval_loss":   [],
                **extra_metrics,
            }
        else:
            if self.text_representer.num_labels == 2:
                if self.args.sliding_window:
                    training_progress_scores = {
                        "global_step": [],
                        "tp":          [],
                        "tn":          [],
                        "fp":          [],
                        "fn":          [],
                        "mcc":         [],
                        "train_loss":  [],
                        "eval_loss":   [],
                        **extra_metrics,
                    }
                else:
                    training_progress_scores = {
                        "global_step": [],
                        "tp":          [],
                        "tn":          [],
                        "fp":          [],
                        "fn":          [],
                        "mcc":         [],
                        "train_loss":  [],
                        "eval_loss":   [],
                        "auroc":       [],
                        "auprc":       [],
                        **extra_metrics,
                    }
            elif self.text_representer.num_labels == 1:
                training_progress_scores = {
                    "global_step": [],
                    "train_loss":  [],
                    "eval_loss":   [],
                    "mcc":         [],
                    **extra_metrics,
                }
            else:
                training_progress_scores = {
                    "global_step": [],
                    "mcc":         [],
                    "train_loss":  [],
                    "eval_loss":   [],
                    **extra_metrics,
                }

        return training_progress_scores

    def save_model(
            self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None
    ):
        pass

        # if not output_dir:
        #     output_dir = self.args.output_dir
        # makedirs(output_dir, exist_ok=True)
        #
        # if model and not self.args.no_save:
        #     # Take care of distributed/parallel training
        #     model_to_save = model.module if hasattr(model, "module") else model
        #     model_to_save.save_pretrained(output_dir)
        #     self.tokenizer.save_pretrained(output_dir)
        #     save(self.args, join(output_dir, "training_args.bin"))
        #     if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
        #         save(
        #             optimizer.state_dict(), join(output_dir, "optimizer.pt")
        #         )
        #         save(
        #             scheduler.state_dict(), join(output_dir, "scheduler.pt")
        #         )
        #     self.save_model_args(output_dir)

        # if results:
        #     output_eval_file = join(output_dir, "eval_results.txt")
        #     with open(output_eval_file, "w") as writer:
        #         for key in sorted(results.keys()):
        #             writer.write("{} = {}\n".format(key, str(results[key])))

    def save_model_args(self, output_dir):
        makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = ClassificationArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.text_representer.named_parameters()]


def format_df_cls(df: pd.core.frame.DataFrame):
    """ Converts input to proper format for simpletransformer. """
    ## Consolidate labels:
    df['labels'] = df[df.columns[1:]].values.tolist()
    ## Keep required columns only:
    df = df[['text', 'labels']].copy()
    return df


def process_mkaggle_data(kl_name, process_labels=False):
    kl = pd.read_csv(kl_name, index_col='id')

    ## Copy English texts for empty (NAN) values:
    kl['original'] = np.where(kl['original'].isnull(), kl['message'], kl['original'])

    ## Drop unnecessary columns:
    kl = kl.drop(columns=['split', 'genre', 'message'])

    ## Rename columns:
    kl.rename(columns={'original': 'text'}, inplace=True)

    if process_labels:
        kl = format_df_cls(kl)

    return kl


def process_mkaggle_datas():
    filenames = ['mKaggle_train', 'mKaggle_val', 'mKaggle_test']

    for filename in filenames:
        kl = process_mkaggle_data(join(dataset_dir, filename + '.csv'))
        kl.to_csv(join(dataset_dir, filename + '_processed.csv'))


def process_ecuador_data(ec_name):
    ec = pd.read_csv(ec_name, index_col='id')

    ec.drop(columns=['screen_name', 'url', 'timestamp', 'choose_one_category',
                     'choose_one_category_a1', 'choose_one_category_a2', 'choose_one_category_a3'], inplace=True)

    ec.crisis_related.replace(to_replace=['no', 'yes'], value=[0, 1], inplace=True)

    ec.rename(columns={'crisis_related': 'labels'}, inplace=True)

    return ec


from Data_Handlers.create_datasets import split_csv_dataset


def process_ecuador_datas():
    filenames = ['ecuador-en', 'ecuador-es']

    for filename in filenames:
        ec = process_ecuador_data(join(dataset_dir, filename + '.csv'))
        new_name = join(dataset_dir, filename + '_processed.csv')
        ec.to_csv(new_name)

        ec_train, ec_val, ec_test = split_csv_dataset(dataset_name=filename + '_processed.csv',
                                                      dataset_dir=dataset_dir, frac=0.6)
        ec_train.to_csv(join(dataset_dir, filename + '_train.csv'))
        ec_val.to_csv(join(dataset_dir, filename + '_val.csv'))
        ec_test.to_csv(join(dataset_dir, filename + '_test.csv'))


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
    model_args.save_steps = 100000
    model_args.reprocess_input_data = True
    model_args.evaluate_during_training_steps = 100000
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


datanames = {
    'mKaggle': ['mKaggle_train', 'mKaggle_val', 'mKaggle_test', 'mKaggle_source', 'mKaggle_target'],
    'ecuador': ['ecuador_train', 'ecuador_val', 'ecuador_test', 'ecuador_source', 'ecuador_target'],
    'NEQ':     ['NEQ_train', 'NEQ_val', 'NEQ_test', 'NEQ_source', 'NEQ_target'],
    'QFL':     ['QFL_train', 'QFL_val', 'QFL_test', 'QFL_source', 'QFL_target'],
    'fire16':  ['fire16_train', 'fire16_val', 'fire16_test', 'fire16_source', 'fire16_target'],
    'smerp17': ['smerp17_train', 'smerp17_val', 'smerp17_test', 'smerp17_source', 'smerp17_target'],
}


def get_input_file_paths(data_name=cfg['data']['name'], data_dir=dataset_dir):
    # train_name, val_name, test_name, source_name, target_name = datanames[data_name]
    # source_name, target_name = data_name.split("-")
    source_name, target_name = data_name,data_name
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


def get_bert2gcn_token_map(bert_vocab_s2i, gcn_vocab_s2i, oov_id=0):
    bert2gcn_id_map = {}

    for token, id in bert_vocab_s2i.items():
        if token in gcn_vocab_s2i:
            bert2gcn_id_map[id] = gcn_vocab_s2i[token]
        else:
            bert2gcn_id_map[id] = oov_id

    return bert2gcn_id_map


def get_bert2gcn_id_mapped_examples(bert_examples_ids, bert_vocab_i2s, gcn_vocab_s2i,
                                    special_ids, oov_id=0):
    mapped_examples = []
    for example_ids in bert_examples_ids:
        mapped_example = []
        for token_id in example_ids:
            token_id = token_id.item()
            ## Ignore special token ids:
            if token_id in special_ids:
                mapped_example.append(oov_id)
                # ## Stop if reached pad id; nothing to process after pad:
                # if token_id == 0:
                #     break
                continue

            if bert_vocab_i2s[token_id] in gcn_vocab_s2i:
                mapped_example.append(gcn_vocab_s2i[bert_vocab_i2s[token_id]])
            else:
                mapped_example.append(oov_id)

        mapped_examples.append(tensor(mapped_example))

    return stack(mapped_examples)


def pairwise_cosine_sim_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def get_embs_adj(X, threshold=0.8, to_sparse=True):
    embs_adj = pairwise_cosine_sim_torch(X)
    ## Remove diagonal elements:
    embs_adj = embs_adj.fill_diagonal_(0.)
    if threshold is not None:
        embs_adj[embs_adj <= threshold] = 0.

    embs_adj = normalize(embs_adj, p=1, dim=1)

    ## Consider upper diagonal portion only
    ## diagonal=1 considers values above diagonal
    # embs_adj = triu(embs_adj, diagonal=1)

    ## Convert to sparse format:
    if to_sparse:
        embs_adj = embs_adj.to_sparse()

    return embs_adj


def setup_graph_component(args, clf, train_df, train_name,
                          unlabelled_source_name, unlabelled_target_name):
    if args.model_type == 'bert' or args.model_type == 'mbert' or args.model_type == 'glen_bert':
        init_embs = deepcopy(clf.text_representer.bert.embeddings.word_embeddings.weight.data)
    elif args.model_type == 'roberta' or args.model_type == 'glen_roberta'\
            or args.model_type == 'xlmroberta' or args.model_type == 'glen_xlmroberta':
        init_embs = deepcopy(clf.text_representer.roberta.embeddings.word_embeddings.weight.data)
    else:
        raise NotImplementedError(f"Model type [{args.model_type}] is not supported.")

    global_token_graph, A, X, joint_vocab = construct_token_graph(
        init_embs, clf.tokenizer.vocab, train_name, unlabelled_source_name,
        unlabelled_target_name, train_df=train_df, tokenizer=clf.tokenizer,
        data_dir=dataset_dir)

    ## Row-normalize adjacency matrix:
    A = normalize(A.to_dense(), p=1, dim=1).to_sparse()

    if args.multilingual:
        A_embs = get_embs_adj(X, threshold=0.5, to_sparse=True)
        ## TODO: Row-normalize both matrices
        A = add(A, A_embs)
        # A = A_embs + pow(A_embs, A)
        A_embs = None

    clf.tokenizer.bert_vocab_i2s = dict((value, key) for key, value in
                                        clf.tokenizer.vocab.items())

    clf.joint_vocab = joint_vocab

    clf.bert2gcn_id_map = get_bert2gcn_token_map(clf.tokenizer.vocab,
                                                 joint_vocab['str2idx_map'])

    clf.A = A
    clf.A = clf.A.to(clf.device)
    clf.X = X
    clf.X = clf.X.to(clf.device)

    return clf


def main(args, multi_label=cfg['data']['multi_label'], exp_name=''):
    train_name, val_name, test_name, train_path, val_path, test_path,\
    unlabelled_source_name, unlabelled_target_name, unlabelled_source_path,\
    unlabelled_target_path, source_name, target_name = get_input_file_paths(
        data_dir=dataset_dir)

    train_df, val_df, test_df = read_data(
        train_name=train_name, val_name=val_name, test_name=test_name,
        data_dir=dataset_dir, train_portion=args.train_portion,
        format_input=multi_label)

    # if train_name != 'mKaggle':
    #     train_df.labels = train_df.labels.apply(lambda x: x[1:])
    #     val_df.labels = val_df.labels.apply(lambda x: x[1:])
    #     test_df.labels = test_df.labels.apply(lambda x: x[1:])

    if multi_label:
        num_classes = len(train_df.labels.to_list()[0])
        # num_classes = len(train_df.labels.to_list())
    else:
        if type(train_df.labels.to_list()[0]) is list:
            num_classes = len(train_df.labels.to_list()[0])
        else:
            num_classes = cfg['data']['num_classes']
    model_args = set_model_args(n_classes=num_classes, lr=args.lr,
                                num_epoch=args.num_train_epochs, in_dim=768,
                                train_all_bert=True)

    model_args.multi_label = multi_label

    clf = ClassificationModel(
        model_type=args.model_type, model_name=args.model_name,
        use_graph=args.use_graph, estimate_importance=args.estimate_importance,
        num_labels=num_classes, args=model_args, exp_name=exp_name,
        use_cuda=args.use_cuda, cuda_device=cuda_device)

    txt = "maybe it's time we form a group of digital disaster responders who mobilize #Ushahidi for #disasters"
    print(clf.tokenizer.tokenize(txt))

    if args.use_graph:
        clf = setup_graph_component(args, clf, train_df, train_name, unlabelled_source_name, unlabelled_target_name)

    eval_dicts = {}
    eval_dicts[target_name] = test_df
    extra_names = []
    if source_name != target_name:
        extra_names.append(source_name)
    for extra_name in extra_names:
        eval_df = read_csv(data_dir=dataset_dir, data_file=extra_name + "_test")

        # if extra_name != 'mKaggle':
        #     eval_df.labels = eval_df.labels.apply(lambda x: x[1:])
        if multi_label and not extra_name.startswith('mkaggle'):
            eval_df = format_df_cls(eval_df)
        elif type(eval_df.labels.iloc[0]) == str:
            eval_df.labels = eval_df.labels.map(ast.literal_eval)
        eval_dicts[extra_name] = eval_df

    class_weights = None
    if train_name.startswith('mkaggle'):
        labels_hot_np = np.array(train_df.labels.to_list())
        class_weights = calculate_class_weights(labels_hot_np=labels_hot_np)
    
    clf.eval_model(eval_df=val_df, multi_label=multi_label, weighted_f1=weighted_f1)

    logger.info(clf.train_model(train_df, eval_dfs=eval_dicts, class_weights=class_weights,
                                multi_label=multi_label, weighted_f1=weighted_f1))

    # extra_names = [args.source_name + "_test", ]
    # for extra_name in extra_names:
    #     extra_df = read_csv(data_dir=dataset_dir, data_file=extra_name)
    #     clf.eval_model(eval_df=extra_df, multi_label=multi_label, weighted_f1=weighted_f1)

    logger.info("Execution Completed")


def calculate_class_weights(labels_hot_np: np.ndarray):
    """ Calculates class weights with 'balanced' mode.

    NOTE: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    NOTE: Adds extra 1 to handle 0 count

    :param labels_hot_np: Multi-Hot vector of numpy ndarray.

    Formula to calculate this is:

        w_j = n_samples / (n_classes * n_samples_j)

        Here,

        w_j is the weight for each class(j signifies the class)
        n_samples is the total number of samples or rows in the dataset
        n_classes is the total number of unique classes in the target
        n_samples_j is the total number of rows of the respective class
        For our heart stroke example:

        n_samples = 43400, n_classes = 2 (0 and 1), n_sample_0 = 42617, n_samples_1 = 783

        Weight for class 0:

        w0 = 43400/(2 * 42617) = 0.509

        Weight for class 1:

        w1 = 43400/(2 * 783) = 27.713

    """
    weights = np.zeros(labels_hot_np.shape[1])
    for i in range(labels_hot_np.shape[1]):
        # logger.debug((labels_hot_np.shape[0], labels_hot_np.shape[1], sum(labels_hot_np[:, i])))
        cls_count = sum(labels_hot_np[:, i])
        if sum(labels_hot_np[:, i]) > 0:
            weights[i] = labels_hot_np.shape[0] / (labels_hot_np.shape[1] * cls_count)
        else:
            weights[i] = 0.

    return from_numpy(weights)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("-g", "--use_graph", default=cfg['data']['use_graph'], type=bool,
                        help="False is for without graph component.")
    parser.add_argument("-ei", "--estimate_importance", default=cfg['data']['estimate_importance'], type=str,
                        help="'new' is for Scaled Dot Product Attention; 'none' to disable.")
    parser.add_argument("-en", "--exp_name", default='bert', type=str)
    parser.add_argument("-c", "--use_cuda", default=True, type=bool)
    parser.add_argument("-m", "--model_name", default=cfg['transformer']['model_name'], type=str)
    parser.add_argument("-mt", "--model_type", default=cfg['transformer']['model_type'], type=str)
    parser.add_argument("-e", "--num_train_epochs", default=cfg['transformer']['num_epoch'], type=int)
    parser.add_argument("-d", "--dataset_name", default=cfg['data']['name'], type=str)
    parser.add_argument("-ml", "--multilingual", default=cfg['data']['multilingual'], type=bool)
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
            logger.warning(f'Run for Experiment: [{args.exp_name}]')
            main(args, multi_label=cfg['data']['multi_label'],
                 exp_name=exp_name)

            # args.estimate_importance = 'none'
            # # args.exp_name = args.exp_name +'_ei_'+str(args.estimate_importance)
            # logger.warning(f'Run for estimate_importance: [{args.estimate_importance}]')
            # main(args, multi_label=cfg['data']['multi_label'],
            #      exp_name=exp_name + '_estimate_importance_' + str(args.estimate_importance))

            # args.use_graph = not args.use_graph
            # # args.exp_name = args.exp_name + '_use_graph_' + str(args.use_graph)
            # logger.warning(f'Run for use_graph: [{args.use_graph}]')
            # main(args, multi_label=cfg['data']['multi_label'],
            #      exp_name=exp_name + '_use_graph_' + str(args.use_graph))


    # exp_name = 'defaultlr_'
    # train_portions = cfg['data']['train_portions']
    # for t_portion in train_portions:
    #     args.train_portion = t_portion
    #     args.exp_name = exp_name + 'train_portion_[' +\
    #                     str(args.train_portion) + ']_'
    #     logger.info(f'Run for TRAIN portion: [{args.train_portion}]')
    #     run_main()

    exp_name = ''
    lrs = cfg['transformer']['lrs']
    train_portions = cfg['data']['train_portions']
    for t_portion in train_portions:
        args.train_portion = t_portion
        # exp_name = exp_name + 'train_portion_[' + str(args.train_portion) + ']_'
        for lr in lrs:
            args.lr = lr
            exp_name = 'dataset: [' + str(args.dataset_name) + '], ' +\
                       'lr: [' + str(args.lr) + '], ' +\
                       'portion: [' + str(args.train_portion) + '], '

            logger.info(f'Run for LR: [{args.lr}] Portion: [{args.train_portion}]')
            run_main()

    logger.info(f"Execution completed for {args.seed_count} SEEDs.")

# Experiments:
# 1. Run XLM-R and GLEN-XLM on datasets
# 2. Run all experiments without importance
# 3. Add more data during graph construction and effect on performance
# 4. Update GNN with GAT and GraphSage
# 5. Zero-Shot multilingual experiments
# Manual evaluation over knn by Word2Vec and BERT embes
# 2. Initialize token graph vectors with trained Word2Vec vectors.
# 3. Evaluate importance estimator by checking important scores for stopwords are less.
# 4. Add qualitative examples where BERT fails.
# 5. Qualitative investigation for relation between spurious connections and importance score.
# 6. motivation conflict between importance estimator and higher weightage to co-occurrence adj; row-normalize;
