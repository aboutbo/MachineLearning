# MachineLearning
## 入门机器学习，使用sk-learn对数据使用多种有监督学习算法，识别撞库IP并比对效果
### 1. 数据来源
- web js 埋点采集浏览器特征，包括IP,UA,输入框焦点获取,cookie enable,java enable,鼠标位置,屏幕大小,时间,reffer等
### 2. 数据打标
- 依赖于微步的IP信誉库，黑IP标记为1，白IP标记为0
### 3. 特征选取
- 选取UA,登录页打开方式,输入框焦点获取,cookie enable,java enable,鼠标位置，屏幕大小
### 4. 特征处理
- UA：使用2-gram，生成特征向量。
- 其他作为离散值，处理为0，1，2等
### 5. 模型训练
- 样本数量总数：65000
- 黑IP样本：49493
- 白IP样本：15507
- 验证方式：10折交叉验证
- 样本分成10份，每次9个子样本用来训练，1个子样本用来验证，重复10次。

#### - LogisticRegression模型
![](file:///MachineLearning/images/LR.png)
#### -
