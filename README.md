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
![](https://github.com/aboutbo/MachineLearning/raw/master/images/LR.png)
#### - DecisionTree模型
![](https://github.com/aboutbo/MachineLearning/raw/master/images/DT.png)
#### - KNN模型
![](https://github.com/aboutbo/MachineLearning/raw/master/images/KNN.png)
#### - NaiveBayes模型
![](https://github.com/aboutbo/MachineLearning/raw/master/images/NB.png)
#### - RandomForest模型
![](https://github.com/aboutbo/MachineLearning/raw/master/images/RF.png)
#### - SVM模型
![](https://github.com/aboutbo/MachineLearning/raw/master/images/SVM.png)

### 6. 总结
- 缺乏前期对数据的统计分析
- 没有对数据进行清洗，比如去重，去除异常值
- 训练模型没有保存
- 特征处理的方式有待改进

## 对第一次过程进行改进
### 1. 数据统计
- 对各个字段的值进行统计，发现一些字段在正负样本中取值无差别，其他一些字段不同取值的比例在
正负样本中基本相同
### 2. 数据去重
- 根据一些字段进行去重
### 3. 去除异常值
- 数据统计过程中发现一个全0的异常值
### 4. 特征构建
- UA：使用n-gram，包括.和/符号
- 离散特征改为one-hot编码
- 连续值进行标准化
### 5. 模型训练
- 改变验证方法：验证方法：样本60%为训练样本，40%为测试样本，计算
Recall rate召回率：TP/(TP+FN)
Precision rate准确率：TP/(TP+FP)
Accuracy rate：整体准确率(TP+TN)/ALL
Confusion matrix混淆矩阵
F1-score 准确率和召回率的调和值
AUC ：ROC曲线面积，值越大模型性能越好
#### - LogisticRegression模型
![](https://github.com/aboutbo/MachineLearning/raw/master/images/LR2.png)
#### - DecisionTree模型
![](https://github.com/aboutbo/MachineLearning/raw/master/images/DT2.png)
#### - KNN模型
![](https://github.com/aboutbo/MachineLearning/raw/master/images/KNN2.png)
#### - NaiveBayes模型
![](https://github.com/aboutbo/MachineLearning/raw/master/images/NB2.png)
#### - RandomForest模型
![](https://github.com/aboutbo/MachineLearning/raw/master/images/RF2.png)
#### - SVM模型
![](https://github.com/aboutbo/MachineLearning/raw/master/images/SVM2.png)
#### - GradientBoosting模型
![](https://github.com/aboutbo/MachineLearning/raw/master/images/GB.png)

### 6. 效果排序
- 根据AUC值排序
RandomForest > GradientBoosting > KNN > LogisticRegression > DecisionTree > SVM > NaiveBayes

- 根据f1_score排序
GradientBoosting > SVM > LogisticRegression > KNN > RandomForest > DecisionTree > NaiveBayes

### 7. 参数调优
- 针对RandomForest使用网格搜索方法尝试调优，AUC提高，precision下降，recall提高
![](https://github.com/aboutbo/MachineLearning/raw/master/images/RF3.png)


### 8. 总结
- 数据打标完全依赖与微步IP信誉库，打标结果有问题
- 特征选择方式有问题，并且没有对特征进行降维
