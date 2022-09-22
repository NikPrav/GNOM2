Effective Utilization of Labeled Data from Related Tasks Using Graph Contrastive Pre-Training
=====

Abstract
-----

Difficulty in obtaining sufficient labeled data for specialized domains has resulted in significant advances in important areas such as pre-training techniques, large language models and transfer learning. These techniques help in reducing dependence on labeled data. The pre-dominant approach of training a model on large unlabeled corpora and then fine-tuning on task specific data is successful but not without its shortcomings. Especially, without sufficient task specific data the fine-tuning process might capture biases present in small datasets and lead to poor generalization. Pre-training has been largely studied in an unsupervised setting. However, oftentimes labeled data from related tasks which share label semantics with current task is available. We hypothesize that using this labeled data effectively can lead to better generalization on current task. In this paper, we propose a novel way to effectively utilize labeled data from related tasks with a graph based supervised contrastive learning formulation. Our formulation results in an embedding space where tokens with high/low probability of belonging to same class are near/further-away from one another. We also develop detailed theoretical insights which serve as a motivation for our method. In our experiments with 13 datasets, we show our method outperforms pre-training schemes by 2.5% and also example-level contrastive learning based formulation by 1.8% on average. In addition, we show cross-domain effectiveness of our method in a zero-shot setting by 3.91% on average. Lastly, we also demonstrate our method can be used as a noisy teacher in a knowledge distillation setting to significantly improve performance of transformer based models in low labeled data regime by 4.57% on average.

## Directory Structure
- `Layers`: Contains all model codes (e.g. LSTM, MLP, GCN etc)
- `Metrics`: Calculates metrics
- `stf_classification`: BERT related codes
- `Pretrain`: GCPT pre-training related codes
- `Disaster_Models`: Other neural network codes (e.g. CNN, XML-CNN, DenseCNN, etc)
- `config.py`: contains all configuration details


## Requirements
Requirement files are located in:

**Conda**:

[requirements/conda_requirements.txt](requirements/conda_requirements.txt)

**Pip**:

[requirements/pip_requirements.txt](requirements/pip_requirements.txt)

## Data Details

### Data Locations

**CrisisLexT6** [https://crisislex.org/data-collections.html#CrisisLexT6](https://crisislex.org/data-collections.html#CrisisLexT6)

1. Download the data
2. Convert label names from "on-topic" and "off-topic" to 1 and 0 respectively.
3. Remove quotations from tweet_id column.

**Amazon Reviews Sentiment** 

1. Download the data
2. Merge *_positive and *_negative files into a single csv file with texts from 
   *_positive anootated with 1 and *_negative annotated with 0
3. Add auto-incremented row id


### Data Format

Traning data format: DataFrame with columns `["id", "text", "labels"]`


## Experiments

- All hyper-parameter and path details are set in `config.py` file

- Run normal experiment (Table 2 & 3) with `python main.py -exp_type 'gcpd_normal'`

- Run Example-level Contrastive Learning (ECL) experiment (Table 4) with 
  `python 
  main.py -exp_type 'gcpd_ecl'`

- Run Related-task Data Fine-tuning (RDF) experiment (Table 5 & 6) with `python 
  main.py -exp_type 
  'gcpd_alltrain'`

- Run Zero-Shot experiments (Table 7 & 8) with `python main.py -exp_type 
  'gcpd_zeroshot'`

- Run Knowledge Distillation (BERT-0.5 & BERT-KD; Table 9 & 10) experiment with 
  `python main_kd.py`
