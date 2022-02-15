#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
准备模型验证的数据
1. 美妆库10W数据 美妆领域NER
2. 美妆库10W数据 故事会2.0NER
'''


# In[28]:


import pandas as pd
import pickle
import json


# In[7]:


meizhuang = pd.read_csv('meizhuangdata_meizhuang_domain_ner.csv')
story = pd.read_csv('meizhuangdata_gushihui_domain_ner.csv')


# In[8]:


meizhuang.info()


# In[9]:


story.info()


# In[40]:


meizhuang_res_dict = dict()
def split_func(x):
    return x.split('|')
for col in meizhuang.columns:
    if col=='唯一ID':
        continue
    else:
        tmp_set = set()
        f_l = list(filter(lambda x: not pd.isnull(x), meizhuang[col].tolist()))
        for ii in f_l:
            for jj in ii.split('|'):
                tmp_set.add(jj)
#         final_list = set(sum(split_list, []))
        tmp_set = list(tmp_set)
        meizhuang_res_dict[col] = tmp_set


# In[42]:


# meizhuang_res_dict['产品']


# In[31]:


story_res_dict = dict()
for idx, row in story.iterrows():
    
#     if not row['内容_新概念高级版_洗衣机_新概念提取'].startswith('{'):
    if pd.isna(row['内容_新概念高级版_洗衣机_新概念提取']):
        continue
    else:
        jsn = json.loads(row['内容_新概念高级版_洗衣机_新概念提取'])
        for dim, words in jsn.items():
            if dim not in story_res_dict:
                story_res_dict[dim] = set()
            for word in words:
                story_res_dict[dim].add(word)


# In[32]:


story_res_dict['产品痛点']


# In[36]:


for kk, vv in story_res_dict.items():
    story_res_dict[kk] = list(vv)


# In[38]:


# story_res_dict['产品痛点']


# In[39]:


with open('storylab_eval_data.pkl', 'wb') as f:
    pickle.dump(story_res_dict, f)


# In[43]:


with open('meizhuang_eval_data.pkl', 'wb') as f:
    pickle.dump(meizhuang_res_dict, f)


# In[44]:


meizhuang_res_dict.keys()


# In[45]:


story_res_dict.keys()


# In[ ]:




