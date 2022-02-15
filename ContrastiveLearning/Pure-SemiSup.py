# -*- coding: utf-8 -*- 
# Author: Henry.Yuan
# Datetime: 2022/2/14 14:09
# File: Pure-SemiSup.py
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
import os

'''
BatchSemiHardTripletLoss 单模型
'''

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_batch_size = 200

# 读取commonKB和产品归一化表数据
# with open('../Data/ContrastiveLearning/commonKB_final_list.pkl', 'rb') as f:
#     commonKB_data = pickle.load(f)
#
# with open('../Data/ContrastiveLearning/product_final_list.pkl', 'rb') as f:
#     product_name_data = pickle.load(f)
#
# all_data = commonKB_data + product_name_data

with open('../Data/ContrastiveLearning/commonKB_food_product.pkl', 'rb') as f:
    all_data = pickle.load(f) # [[sent1, sent2], [sent3, sent4]]

num_epochs = 20

# train_examples
train_examples = list()
for label, data in enumerate(all_data):
    for cur_text in data:
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

word_embedding_model = models.Transformer('hfl/chinese-roberta-wwm-ext', max_seq_length=32, cache_dir='/data/haobin/model/chinese-roberta-wwm-ext')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=train_batch_size)
train_loss = losses.BatchSemiHardTripletLoss(model=model)

warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

def print_func(score, epoch, steps):
    print(score, epoch, steps)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=100,
          warmup_steps=warmup_steps,
          output_path='/data/haobin/model/pure_SemiSup',
          optimizer_params={'lr': 5e-6, 'eps': 1e-9, 'correct_bias': False},
          patience=10,
          callback=print_func
          )

