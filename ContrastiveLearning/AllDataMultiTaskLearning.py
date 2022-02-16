# -*- coding: utf-8 -*- 
# Author: Henry.Yuan
# Datetime: 2022/2/15 18:34
# File: AllDataMultiTaskLearning.py
# Software: PyCharm
#
"""

"""

from read_data import read_train_data, read_dev_data
from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
import pickle
from sentence_transformers import SentenceTransformer, models
import torch
from torch.utils.data import DataLoader
import json
import random
import math
import os
from torch import nn
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import datetime


# 'ds', 'ONLI', 'STS-B', 'xiaobu'

train_batch_size = 100

# prepare Model
word_embedding_model = models.Transformer('hfl/chinese-roberta-wwm-ext', max_seq_length=32, cache_dir='/data/haobin/model/chinese-roberta-wwm-ext')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], cache_folder='/data/haobin/model/multi_task_cache')


# -----------------  prepare data

# ---------------- ONLI
onli_train_examples = read_train_data('ONLI')
onli_train_dataset = SentencesDataset(onli_train_examples, model)
onli_train_dataloader = DataLoader(onli_train_dataset, shuffle=True, batch_size=train_batch_size)
train_onli_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=3)


# --------------- STS-B
sts_train_examples = read_train_data('STS-B')
sts_train_dataset = SentencesDataset(sts_train_examples, model)
sts_train_dataloader = DataLoader(sts_train_dataset, shuffle=True, batch_size=train_batch_size)
train_sts_loss = losses.CosineSimilarityLoss(model=model)


# --------------- xiaobu
xiaobu_train_examples = read_train_data('xiaobu')
xiaobu_train_dataset = SentencesDataset(xiaobu_train_examples, model)
xiaobu_train_dataloader = DataLoader(xiaobu_train_examples, shuffle=True, batch_size=train_batch_size)
train_xiaobu_loss = losses.OnlineContrastiveLoss(model=model)


# ------------- ds
ds_train_examples = read_train_data('ds')
ds_train_dataset = SentencesDataset(ds_train_examples, model)
ds_train_dataloader = DataLoader(ds_train_dataset, shuffle=False, batch_size=train_batch_size)
train_ds_loss = losses.BatchSemiHardTripletLoss(model=model)


# dev_data
# dev_samples = read_dev_data(name='sts')
# dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')
dev_samples = list()
with open('../Data/ContrastiveLearning/storylab_dev.json', 'r', encoding='utf-8') as fIn:
    for row in fIn.readlines():
        j=json.loads(row)
        dev_samples.append(InputExample(texts=[j['sentence1'], j['sentence2']], label=j['label']))

print('Train data info:\n')
print('OCNLI: {}'.format(len(onli_train_examples)))
print('STS-B: {}'.format(len(sts_train_examples)))
print('Xiaobu: {}'.format(len(xiaobu_train_examples)))

print('# valid samples: {}'.format(len(dev_samples)))




dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

# fit model
num_epochs = 20

warmup_steps = math.ceil(len(ds_train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
print('warmup_steps: {}'.format(warmup_steps))

train_objectives = [(xiaobu_train_dataloader, train_xiaobu_loss), (onli_train_dataloader, train_onli_loss),
                    (ds_train_dataloader, train_ds_loss), (sts_train_dataloader, train_sts_loss)]

# train_objectives = [(xiaobu_train_dataloader, train_xiaobu_loss), (onli_train_dataloader, train_onli_loss),
#                     (sts_train_dataloader, train_sts_loss)]

# train_objectives = [(onli_train_dataloader, train_onli_loss)]

def print_func(score, epoch, steps):
    print(score, epoch, steps)

# output_path = '/data/haobin/'
print('Train start in {}'.format(datetime.datetime.now()))
# model = nn.DataParallel(model)
model.fit(train_objectives=train_objectives,
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=150,
          warmup_steps=warmup_steps,
          output_path='/data/haobin/model/all-data-sup-multi-task-output',
          show_progress_bar=True,
          optimizer_params={'lr': 5e-5, 'eps': 1e-9, 'correct_bias': False},
          callback=print_func,
          patience=10)

print('Train finished in {}'.format(datetime.datetime.now()))
