# -*- coding: utf-8 -*- 
# Author: Henry.Yuan
# Datetime: 2022/2/7 14:47
# File: read_data.py
# Software: PyCharm
#
"""

"""
from sentence_transformers import InputExample
import pickle
import json


def read_dev_data(name):
    dev_samples = list()
    if name == 'ds':
        with open('../Data/ContrastiveLearning/storylab_dev.json', 'r', encoding='utf-8') as fIn:
            for row in fIn.readlines():
                j=json.loads(row)
                dev_samples.append(InputExample(texts=[j['sentence1'], j['sentence2']], label=j['label']))
        return dev_samples
    elif name=='sts':
        with open('../Data/ContrastiveLearning/Chinese-STS-B_dev_data.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                split_line = line.split('\t')
                assert len(split_line) == 3, RuntimeError('Len must be 3')
                if len(split_line[0]) > 32 or len(split_line[0]) > 32:
                    continue
                dev_samples.append(InputExample(texts=split_line[:2], label=float(split_line[-1])/5.0))
        return dev_samples
    else:
        assert 1 == 0, print('name must be ds or sts')


def read_train_data(name, sample_num=None):
    if name == 'ds':
        with open('../Data/ContrastiveLearning/commonKB_final_list.pkl', 'rb') as f:
            commonKB_data = pickle.load(f)

        with open('../Data/ContrastiveLearning/product_final_list.pkl', 'rb') as f:
            product_name_data = pickle.load(f)

        all_data = commonKB_data + product_name_data

        # BatchSemiHard Loss Train examples
        train_examples = list()
        for label, data in enumerate(all_data):
            for cur_text in data.split('|'):
                train_examples.append(InputExample(texts=[cur_text], label=label))

        print('{} get {} train samples'.format(name, len(train_examples)))

        return train_examples

    elif name == 'ONLI':
        label_map = {'entailment':0, 'neutral': 1, 'contradiction': 2}
        # for Softmax Loss
        train_examples = list()
        with open('../Data/ContrastiveLearning/Chinese_OCNLI_train.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                split_line = line.split('\t')
                # print(split_line)
                assert len(split_line) == 3, RuntimeError('Len must be 3')
                if len(split_line[0]) > 32 or len(split_line[0]) > 32:
                    continue
                train_examples.append(InputExample(texts=split_line[:2], label=label_map[split_line[-1]]))

        print('{} get {} train samples'.format(name, len(train_examples)))
        return train_examples

    elif name == 'xiaobu':
        # for OnlineContrastiveLoss
        train_examples = list()
        # label_set = set()
        with open('../Data/ContrastiveLearning/OPPO_xiaobu_train_data.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                split_line = line.split('\t')
                assert len(split_line) == 3, RuntimeError('Len must be 3')
                if len(split_line[0]) > 32 or len(split_line[0]) > 32:
                    continue
                # label_set.add(int(split_line[-1]))
                train_examples.append(InputExample(texts=split_line[:2], label=int(split_line[-1])))

        # print(label_set)
        print('{} get {} train samples'.format(name, len(train_examples)))
        return train_examples

    elif name == 'STS-B':
        # for CosineSimilarityLoss
        train_examples = list()
        with open('../Data/ContrastiveLearning/Chinese-STS-B_train_data.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                split_line = line.split('\t')
                assert len(split_line) == 3, RuntimeError('Len must be 3')
                if len(split_line[0]) > 32 or len(split_line[0]) > 32:
                    continue
                train_examples.append(InputExample(texts=split_line[:2], label=float(split_line[-1])/5.0))

        print('{} get {} train samples'.format(name, len(train_examples)))
        return train_examples

    else:
        raise RuntimeError('name must be in {}'.format(['ds', 'ONLI', 'STS-B', 'xiaobu']))

if __name__ == '__main__':
    read_train_data('xiaobu')
