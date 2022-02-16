# 语义表征调研任务

## 背景

语义模型：用于把文本（短语/句子/文章）映射到向量空间，在这个空间中，语义相近的距离近，语义不同的距离远

## 更新目的

公司已有的语义模型已上线了一定时间，在此期间，

- 数据层面上，码表、归一化等可利用的数据不断积累
- 模型层面上，21年对比学习模型、双塔模型不断刷新SOTA
- 应用层面上，越来越多来自产品的需求，都需要用到语义模型

## 初步实验方案

https://wiki.datastory.com.cn/pages/viewpage.action?pageId=79202220



## 实际实验方案

### 有监督联合训练（公开数据集双塔模型）

1. 数据集
   
   1. 训练集：
      
      1. 公开数据集：OCNLI，OPPO小布对话，Chinese-STS-B
      
      2. 公司内部数据：CommonKB（各行业使用的归一化码表），食品归一化码表，美妆产品名归一化码表（commonKB_food_product.pkl）
   
   2. 验证集：
      
      1. Chinese-STS-B dev set（文本更general）

2. 数据处理
   
   1. 对公开数据集，只保留两个文本长度均小于等于32的句子对
   
   2. 公司内部数据集，合并去重，对每组相似的词，最多只保留其中7个

3. 模型训练
   
   1. 模型结构：双塔模型
   2. round-robin轮询式联合训练
      1. 每个step里，各数据集里取一个batch训练并更新模型参数
      2. 训练集与损失函数：
         1. OCNLI（三分类label: neutral, entailment, contradiction） -  SoftmaxLoss
         2. Chinese-STS-B（离散label，0~5） - CosineSimilarityLoss
         3. OPPO小布对话（离散label, 0,1） -  OnlineContrastiveLoss
   3. 训练策略：使用验证集的cosine similarity做early stopping



## 半监督训练

1. 训练集：公司内部数据集（commonKB_food_product.pkl）

2. 验证集：公司内部数据集（storylab_dev.json）

3. 模型结构：双塔模型

4. 损失函数：BatchSemiHardTriplet



## 实验结果


