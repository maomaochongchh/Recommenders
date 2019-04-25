##### ﻿首次学习推荐系统相关的知识，计划将这个项目中算法相关内容翻译为中文，方便自己理解。

## 一、SAR Single Node on MovieLens

​	对应文件：~/Recommenders/notebooks/00_quick_start/sar_movielens.ipynb

​	SAR是一种基于用户事务历史的快速可扩展自适应推荐算法。它生成可解释/说明的建议，并处理“物品冷启动”和“半用户冷启动”场景。SAR是一种基于邻域的算法，它的目的是为每个用户推荐他可能感兴趣的最热门的条目，比如说客户A看了物品1，客户B同时看了1和2，那么有可能客户A也对物品2感兴趣，则推荐给A。
	SAR推荐最“类似”于用户已有“关联”的项目。如果与一个条目交互的用户也可能与另一个条目交互，那么两个条目是“类似的”。如果用户过去曾与某项进行过交互，则该用户对该项具有“关联”。

#### SAR的优点：

​	1.精度高，便于训练和算法实现简单
	2.快速训练，只需要简单的计算来构建预测时使用的矩阵
	3.快速评分，只涉及与关联向量的相似性矩阵乘法

#### SAR的缺点：

​	1.由于它不使用项目或用户特征，相对于使用这些特征的算法，颇有劣势
	2.需要大量内存，需要创建一个 mxm 的稀疏方阵 (其中m是项数)。这对于许多矩阵分解算法来说也是一个问题
	**3.SAR倾向于隐式评级方案，并不预测分数（没理解）。**

#### 内容补充：

​	SAR = Similiarity Affinity Recommendation
	以下内容参考[博客](https://blog.csdn.net/csdn_47/article/details/88351075)：https://blog.csdn.net/csdn_47/article/details/88351075

##### 	SAR算法流程：

​	1.计算物品相似度矩阵$S$ 
	2.用于评估用户和物品的关系矩阵$A$ 
	3.评价分数为计算$R=A∗S$
	4.可选步骤: 包括时间衰减和移除已经看过的物品

![20190308142609352](C:\Users\windows\Desktop\20190308142609352.png)

##### 计算物品（条目）相似度$S$

​	计算相似度有多种不同的方式，不过都首先要计算物品出现的次数。假设一共有$m$种物品，则得到$m*m$大小的矩阵$C$, $c_{i,j}$代表物品$i$,$j$同时出现在一个人的事物列表中的次数，$C$满足以下条件：

- 对称： $c_{i,j}=c_{j,i}$
- 非负性：$c_{i,j}>=0$
- 单个物品出现的洗漱肯定比两两同时出现的次数大：$c_{i,i},c_{j,j}>=c_{i.j}$

可以通过三种公式计算相似度：

- `Jaccard`: $s_{ij}=\frac{c_{ij}}{(c_{ii}+c_{jj}-c_{ij})}$
- `lift`: $s_{ij}=\frac{c_{ij}}{(c_{ii} \times c_{jj})}$
- `counts`: $s_{ij}=c_{ij}$

##### 计算用品与物品之间的关联矩阵（亲和力分数）$A$

$$a_{ij}=\sum_k w_k \left(\frac{1}{2}\right)^{\frac{t_0-t_k}{T}} $$

其中$a_{i,j}$代表用户$i$对物品$j$的亲和力分数， $k$代表用户$i$对物品$j$有$k$次历史行为，$t_0$代表当前时间，$t_k$代表第$k$次行为发生的时间，时间越靠前，这次行为为亲和力分数计算的贡献越低（权重小），$w_k$ 表示不同类型事件的权重（比如点击，购买的权重不同）。$A$的大小为$n*m$, $n$个用户。

##### $S$与$A$相乘得到$n*m$的矩阵，$R_{i,j}$代表应该将物品j推荐给用户i的分数，排序之前应该将之前出现过的user/item对应位置上的值置为0, 然后进行每个用户的Top_K推荐。

# 二、Wide and Deep Model for Movie Recommendation

​	对应文件：notebooks/00_quick_start/wide_deep_movielens.ipynb

​	参考[博客](http://www.shuang0420.com/2017/03/13/论文笔记%20-%20Wide%20and%20Deep%20Learning%20for%20Recommender%20Systems/)

​	具有广泛交叉列(共现)特征的线性模型可以记忆特征间的相互作用，而深度神经网络(DNN)可以通过低维密集嵌入对离散特征概括特征模式 。[**Wide-and-deep Learning**](https://arxiv.org/abs/1606.07792)  联合训练有宽度的线性模型和深度神经网络，设计将记忆性和概括性结合为一体的推荐系统。

## **《Wide & Deep Learning for Recommender Systems》**

任务：采用Wide &deep构建模型，从而得以更精准做Google Play推荐。 

贡献：

- 提出了Wide & Deep learning框架，将logistic model和forward dnn网络结合起来，既发挥logistic model的优势，又利用dnn和embedding的自动特征组合学习和强泛化能力进行补充，保证记忆能力与泛化能力的均衡。而且将模型整体学习，理论上达到最优

- 在Google Apps推荐的大规模数据上成功应用

- 基于tensorflow开源了代码 

  ### 1.整体架构

  ![img](http://images.shuang0420.com/images/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0%20-%20Wide%20and%20Deep%20Learning%20for%20Recommender%20Systems/1.jpg) 

  ​	由两个部分组成，**检索系统(或者说候选生成系统）** 和 **排序系统(排序网络)**。首先，用 **检索(retrieval)** 的方法对大数据集进行初步筛选，返回最匹配 query 的top100物品列表，这里的检索通常会结合采用 **机器学习模型(machine-learned models)** 和 **人工定义规则(human-defined rules)** 两种方法。从大规模样本中召回最佳候选集之后，再使用 **排序系统** 对每个物品进行算分、排序，分数 P(y|x)，y 是用户采取的行动(比如说下载行为)，x 是特征，包括 :

  - **User features**
    e.g., country, language, demographics
  - **Contextual features**
    e.g., device, hour of the day, day of the week
  - **Impression features**
    e.g., app age, historical statistics of an app

  ![img](http://images.shuang0420.com/images/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0%20-%20Wide%20and%20Deep%20Learning%20for%20Recommender%20Systems/3.jpg)

   **记忆能力理解: Memorization**，主要是学习特征的共性或者说相关性，产生的推荐是和已经有用户行为的物品直接相关的物品。 

  为了达到 Memorization，本文对稀疏的特征采取 cross-product transformation，比如说 AND (user_installed_app=netflix, impression_app=pandora) 这个特征，只有 Netflix 和 Pandora 两个条件都达到了，值才为 1，这类 feature 解释了 co-occurrence 和 target label 之间的关系。 cross-product transformation 的局限在于，对于在训练集里没有出现过的 query-item pair，它不能进行泛化(Generalization)。总结起来，宽度模型的输入是用户安装应用(installation)和为用户展示（impression）的应用间的向量积（叉乘），模型通常训练 one-hot 编码后的二值特征，这种操作不会归纳出训练集中未出现的特征对。

   **泛化能力理解：Generalization**，可以理解为相关性的传递(transitivity)，会学习新的特征组合，来提高推荐物品的多样性，或者说提供泛化能力(Generalization) 。

  ​	泛化往往是通过学习 low-dimensional dense embeddings 来探索过去从未或很少出现的新的特征组合来实现的，通常的 embedding-based model 有 **Factorization Machines(FM)** 和 **Deep Neural Networks(DNN)**。特殊兴趣或者小众爱好的用户，query-item matrix 非常稀疏，很难学习，然而 dense embedding 的方法还是可以得到对所有 query-item pair 非零的预测，这就会导致 over-generalize，推荐不怎么相关的物品。这点和 LR 正好互补，因为 LR 只能记住很少的特征组合。

  ​	为了达到 **Generalization**，我们会引入新的小颗粒特征，如类别特征（安装了视频类应用，展示的是音乐类应用，等等）AND(user_installed_category=video, impression_category=music)，这些高维稀疏的类别特征（如人口学特征和设备类别）映射为低纬稠密的向量后，与其他连续特征（用户年龄、应用安装数等）拼接在一起，输入 MLP 中，最后输入逻辑输出单元。

  ​	一开始嵌入向量(embedding vectors)被随机初始化，然后训练过程中通过最小化损失函数来优化模型。

  总结一下，基于 embedding 的深度模型的输入是 **类别特征(产生embedding)+连续特征**。

  ### 2.联合训练

  ​	对两个模型的输出算 log odds ratio 然后加权求和，作为预测。 Joint Training 同时训练 wide & deep 模型，优化的参数包括两个模型各自的参数以及 weights of sum。

  ### 3.系统实施

  ![6.jpg](http://images.shuang0420.com/images/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0%20-%20Wide%20and%20Deep%20Learning%20for%20Recommender%20Systems/6.jpg) 

  # 三、

  

  

  