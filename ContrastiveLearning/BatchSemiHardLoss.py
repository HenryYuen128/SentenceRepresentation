# -*- coding: utf-8 -*- 
# Author: Henry.Yuan
# Datetime: 2022/1/19 16:25
# File: BatchSemiHardLoss.py
# Software: PyCharm
#
"""

"""
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

'''
BatchSemiHardLoss test

'''


torch.cuda.set_device(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_batch_size = 32

# 读取commonKB和产品归一化表数据
with open('../Data/ContrastiveLearning/commonKB_final_list.pkl', 'rb') as f:
    commonKB_data = pickle.load(f)

with open('../Data/ContrastiveLearning/product_final_list.pkl', 'rb') as f:
    product_name_data = pickle.load(f)

all_data = commonKB_data + product_name_data
num_epochs = 2


# train_examples
train_examples = list()
for label, data in enumerate(all_data):
    for cur_text in data.split('|'):
        train_examples.append(InputExample(texts=[cur_text], label=label))

print(len(train_examples))
# dev examples
dev_samples = list()
with open('../Data/ContrastiveLearning/storylab_dev.json', 'r', encoding='utf-8') as fIn:
    for row in fIn.readlines():
        j=json.loads(row)
        dev_samples.append(InputExample(texts=[j['sentence1'], j['sentence2']], label=j['label']))

print(len(all_data))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

word_embedding_model = models.Transformer('hfl/chinese-roberta-wwm-ext', max_seq_length=48, cache_dir='/data/haobin/model/chinese-roberta-wwm-ext')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], cache_folder='/data/haobin/model/SBERT')
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=train_batch_size)
train_loss = losses.BatchSemiHardTripletLoss(model=model)

warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path='/data/haobin/model/SBERT_demo1',
          optimizer_params={'lr': 5e-8, 'eps': 1e-9, 'correct_bias': False},
          )

