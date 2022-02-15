# -*- coding: utf-8 -*- 
# Author: Henry.Yuan
# Datetime: 2022/2/14 10:27
# File: unsup-SimCSE.py
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
from sentence_transformers.evaluation import BinaryClassificationEvaluator

'''
Multi-task + BatchTripletLoss，后接SimCSE无监督模型
'''


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

train_batch_size = 128

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

num_epochs = 30

# train_examples
train_examples = list()
word_list = list()
for label, data in enumerate(all_data):
    # for cur_text in data:
    pos_sample = random.sample(data, 1)[0]
    word_list.append(pos_sample)
    train_examples.append(InputExample(texts=[pos_sample, pos_sample]))

print('# train samples: {}'.format(len(train_examples)))
with open('/data/haobin/code/Draft/train_samples.pkl', 'wb') as f:
    pickle.dump(word_list, f)


# dev examples
dev_samples = list()
with open('../Data/ContrastiveLearning/storylab_dev.json', 'r', encoding='utf-8') as fIn:
    for row in fIn.readlines():
        j=json.loads(row)
        dev_samples.append(InputExample(texts=[j['sentence1'], j['sentence2']], label=j['label']))

print('# valid samples: {}'.format(len(dev_samples)))


dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

model = SentenceTransformer('/data/haobin/model/semi-sup-tripletLoss')


# train_examples = random.sample(train_examples, 50000)
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

print(len(train_dataloader))
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
print('Warmup steps: {}'.format(warmup_steps))


def print_func(score, epoch, steps):
    print(score, epoch, steps)


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=50,
          warmup_steps=warmup_steps,
          output_path='/data/haobin/model/unsup-SimCSE',
          optimizer_params={'lr': 1e-9, 'eps': 1e-9, 'correct_bias': False},
          callback=print_func,
          patience=10
          )

