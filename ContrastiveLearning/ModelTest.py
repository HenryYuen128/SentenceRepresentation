# -*- coding: utf-8 -*- 
# Author: Henry.Yuan
# Datetime: 2022/2/14 14:34
# File: ModelTest.py
# Software: PyCharm
#
"""

"""
import torch
import os
from sentence_transformers import SentenceTransformer, SentencesDataset, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import pickle
import random
import numpy as np
import pandas as pd
random.seed(2)

'''
模型对比测试
'''


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

multi_task_triplet_model = SentenceTransformer('/data/haobin/model/semi-sup-tripletLoss') # 联合训练+tripletLoss模型
multi_task_model = SentenceTransformer('/data/haobin/model/sup-multi-task-output') # 单联合训练
pure_triplet_model = SentenceTransformer('/data/haobin/model/pure_SemiSup') # 单tripletLoss

model_dict = {'multi-task_triplet': multi_task_triplet_model, 'multi-task': multi_task_model, 'pure_triplet': pure_triplet_model}


# sentences = ['要摆脱毛毛', '汗毛这么长']

# sentences2 = ['汗毛真多的有点多', '腿毛太多了', '毛多太烦人了']

#Sentences are encoded by calling model.encode()
# sentence_embeddings = multi_task_triplet_model.encode(sentences)
# sentence_embeddings2 = multi_task_triplet_model.encode(sentences2)
# sentence_embeddings3 = pure_triplet_model.encode(sentences)

# cos_sim = util.cos_sim(sentence_embeddings, sentence_embeddings2)

# print(cos_sim)


with open('../Data/ContrastiveLearning/meizhuang_eval_data.pkl', 'rb') as f:
    meizhuang_ner = pickle.load(f)
#
with open('../Data/ContrastiveLearning/storylab_eval_data.pkl', 'rb') as f:
    story_ner = pickle.load(f)


def coherence_eval(model, sample_sentences, sample_exclude, topNlist=[0,1,2,99], sample_num=50):

    # sample_sentences = random.sample(sentences, sample_num)
    # sample_exclude = list(set(sentences) - set(sample_sentences))

    #Encode all sentences
    embeddings = model.encode(sample_sentences)
    embeddings_ex = model.encode(sample_exclude)

    #Compute cosine similarity between all pairs
    cos_sim = util.cos_sim(embeddings, embeddings_ex)

    #Add all pairs to a list with their cosine similarity score

    all_sentence_combinations = {sent: list() for sent in sample_sentences}
    for i in range(len(cos_sim)):
        cur_sim_list = cos_sim[i]
        sort_sim_idx_list = np.argsort(-cur_sim_list)
        sort_sim_list = cur_sim_list[sort_sim_idx_list]
        for n in topNlist:
            cur_sent = sample_sentences[i]
            # kk = sort_sim_list[n].numpy().tolist()
            tmp_item = [sample_exclude[sort_sim_idx_list[n]], str(round(sort_sim_list[n].numpy().tolist(), 2))]
            all_sentence_combinations[cur_sent].append(tmp_item)

    return all_sentence_combinations




#----------------------------------- 模型测试

# with open('美妆日化ner功效维度.txt', 'w', encoding='utf-8') as f:
sentences = story_ner['自我描述']
sample_sentences = random.sample(sentences, 50)
sample_exclude = list(set(sentences) - set(sample_sentences))
model_res_dict = {'word': sample_sentences}

for model_name, model in model_dict.items():
    tmp_res_dict = coherence_eval(model, sample_sentences, sample_exclude)
    res_list = list()

    for ii in tmp_res_dict.values():
        tmp_res_list = list()
        for kk in ii:
            tmp_res_list.append(','.join(kk))
        concat_str = '|'.join(tmp_res_list)
        res_list.append(concat_str)
    model_res_dict[model_name] = res_list

pd.DataFrame(model_res_dict).to_excel('自我描述.xlsx', index=False)
