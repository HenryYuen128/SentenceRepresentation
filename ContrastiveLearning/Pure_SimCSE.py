# -*- coding: utf-8 -*- 
# Author: Henry.Yuan
# Datetime: 2022/2/7 14:04
# File: Pure_SimCSE.py
# Software: PyCharm
#
"""

"""

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses
from torch.utils.data import DataLoader
import pickle
import torch
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer, SentencesDataset, losses
import json
import random
import os

'''
单无监督模型(SimCSE)
'''

# torch.cuda.set_device(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_batch_size = 128

# 读取commonKB和产品归一化表数据
# with open('../Data/ContrastiveLearning/commonKB_final_list.pkl', 'rb') as f:
#     commonKB_data = pickle.load(f)
#
# with open('../Data/ContrastiveLearning/product_final_list.pkl', 'rb') as f:
#     product_name_data = pickle.load(f)
#
# all_data = commonKB_data + product_name_data

num_epochs = 1

with open('../Data/ContrastiveLearning/commonKB_food_product.pkl', 'rb') as f:
    all_data = pickle.load(f) # [[sent1, sent2], [sent3, sent4]]

num_epochs = 1

# train_examples
train_examples = list()
for label, data in enumerate(all_data):
    # for cur_text in data:
    pos_sample = random.sample(data, 1)[0]
    train_examples.append(InputExample(texts=[pos_sample, pos_sample]))


print(len(train_examples))
# train_examples = random.sample(train_examples, 10000)

# dev examples
dev_samples = list()
with open('../Data/ContrastiveLearning/storylab_dev.json', 'r', encoding='utf-8') as fIn:
    for row in fIn.readlines():
        j=json.loads(row)
        dev_samples.append(InputExample(texts=[j['sentence1'], j['sentence2']], label=j['label']))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

word_embedding_model = models.Transformer('hfl/chinese-roberta-wwm-ext', max_seq_length=32, cache_dir='/data/haobin/model/chinese-roberta-wwm-ext')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], cache_folder='/data/haobin/model/SimCSE')
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

def print_func(score, epoch, steps):
    print(score, epoch, steps)


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=50,
          warmup_steps=warmup_steps,
          output_path='/data/haobin/model/Pure-SimCSE',
          optimizer_params={'lr': 1e-9},
          show_progress_bar=True,
          patience=5,
          callback=print_func
          )




