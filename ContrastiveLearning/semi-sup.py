# -*- coding: utf-8 -*- 
# Author: Henry.Yuan
# Datetime: 2022/2/11 17:14
# File: semi-sup.py
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
Multi-task模型后接BatchSemiHardLoss模型
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

num_epochs = 100



# train_examples
train_examples = list()
for label, data in enumerate(all_data):
    for cur_text in data:
        train_examples.append(InputExample(texts=[cur_text], label=label))

print('# train samples: {}'.format(len(train_examples)))

# dev examples
dev_samples = list()
with open('../Data/ContrastiveLearning/storylab_dev.json', 'r', encoding='utf-8') as fIn:
    for row in fIn.readlines():
        j=json.loads(row)
        dev_samples.append(InputExample(texts=[j['sentence1'], j['sentence2']], label=j['label']))

print('# valid samples: {}'.format(len(dev_samples)))


dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

model = SentenceTransformer('/data/haobin/model/sup-multi-task-output') # 加载已训练好的Multi-task模型


train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=train_batch_size)
train_loss = losses.BatchSemiHardTripletLoss(model=model)

print(len(train_dataloader))
warmup_steps = int(len(train_dataloader) * num_epochs//2 * 0.1)
print('Warmup steps: {}'.format(warmup_steps))


def print_func(score, epoch, steps):
    print(score, epoch, steps)


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=300,
          warmup_steps=warmup_steps,
          output_path='/data/haobin/model/semi-sup-tripletLoss',
          optimizer_params={'lr': 2e-5, 'eps': 1e-9, 'correct_bias': False},
          callback=print_func,
          patience=5
          )

