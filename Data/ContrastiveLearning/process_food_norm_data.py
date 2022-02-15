#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import pickle
import random
import datetime


# In[ ]:


'''
食品归一化
、commonKB、产品名数据合并
'''


# In[33]:


# 食品归一化数据
df = pd.read_csv('food_norm_data.csv')


# In[34]:


df.head()


# In[35]:


df = df[df['tab']=='归一化']


# In[36]:


df.info()


# In[ ]:





# In[37]:


# commonKB_data


# In[38]:


df['word_num'] = df['wordList'].apply(lambda x: len(x.split('|')))


# In[39]:


df.head()


# In[40]:


df['word_num'].max()


# In[41]:


def sample_func(ll):
    return '|'.join(random.sample(ll, 4)) if len(ll) > 4 else '|'.join(ll)


# In[42]:


df['sample_words'] = df['wordList'].apply(lambda x: sample_func(x.split('|')))


# In[43]:


df.head()


# In[44]:


word_list = df['sample_words'].tolist()


# In[45]:


len(word_list)


# In[46]:


with open('food_norm_data.pkl', 'wb') as f:
    pickle.dump(word_list, f)


# In[47]:


with open('commonKB_final_list.pkl', 'rb') as f:
    commonKB_data = pickle.load(f)
with open('product_final_list.pkl', 'rb') as f:
    product_name_data = pickle.load(f)


# In[48]:


len(commonKB_data), len(product_name_data), len(word_list)


# In[49]:


all_data = commonKB_data + product_name_data + word_list


# In[50]:


len(all_data)


# In[51]:


import copy


# In[55]:


all_data[:5]


# In[65]:


# Python
# l = [set([37, 27]), set([14,15]), set([36,27]), set([54,27]), set([36,54]), set([1,2])]
print(datetime.datetime.now())
# l = copy.deepcopy(all_data)
l = [set(i.split('|')) for i in all_data]
# l = [(1,2), (2,3), (5,6)]
pool = set(map(frozenset, l))

groups = []
while len(pool)>0:
    groups.append(set(pool.pop()))
    while True:
        flag = 1
#         print(groups)
#         if len(pool) == -1:
#             break
        for candidate in pool:
#             print(candidate)
            if groups[-1] & candidate:
                groups[-1] |= candidate
                pool.remove(candidate)
                break
        else:
            break
print(datetime.datetime.now())


# In[66]:


len(groups)


# In[67]:


groups[:5]


# In[73]:


len_list = list()
word_list = list()
for ii in groups:
    tmp_len_list = list()
    tmp_word_list = list()
    for kk in ii:
        if len(kk) <= 32:
            tmp_word_list.append(kk)
    if len(tmp_word_list) > 8:
        tmp_word_list = random.sample(tmp_word_list, 8)
    word_list.append(tmp_word_list)
    


# In[74]:


len(word_list)


# In[75]:


word_list[:5]


# In[76]:


with open('commonKB_food_product.pkl', 'wb') as f:
    pickle.dump(word_list, f)

