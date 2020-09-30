# ML_Basics
A collection of ML interview questions.

##Data Preprocessing

**THREE STAGES TO BUILD ML MODELS**

- Design/ develop model + Train
- Test
- Apply/ Scale up for inference

**NLP数据预处理有啥手段? (TOKENIZATION)**

**MISSING/ CORRUPTED VALUE**
- Remove record if not much
- Replace with average or with 0 depending on the data

**DEALING WITH NUMERICAL + CATEGORICAL  FEATURES**

In general, a preferred approach is to convert all your features into standardized continuous features.
- For features that were originally continuous, perform standardization:
x_i = (x_i - mean(x)) / standard_deviation(x).
That is, for each feature, subtract the mean of the feature and then divide by the standard deviation of the feature. An alternative approach is to convert the continuous features into the range [0, 1]:
x_i = (x_i - min(x)) / (max(x) - min(x)).
- For categorical features, perform binarization on them so that each value is a continuous variable taking on the value of 0.0 or 1.0. For example, if you have a categorical variable "gender" that can take on values of MALE, FEMALE, and NA, create three binary binary variables IS_MALE, IS_FEMALE, and IS_NA, where each variable can be 0.0 or 1.0 One-Hot Encodings . You can then perform standardization as in step 1.
Now you have all your features as standardized continuous variables.

**POSITIVE NEGATIVE SAMPLE IMBALANCE**

An imbalanced dataset is one that has different proportions of target categories. For example, a dataset with medical images where we have to detect some illness will typically have many more negative samples than positive samples—say, 98% of images are without the illness and 2% of images are with the illness.
There are different options to deal with imbalanced datasets:
1.	Oversampling or undersampling. Instead of sampling with a uniform distribution from the training dataset, we can use other distributions so the model sees a more balanced dataset.
2.	Data augmentation. We can add data in the less frequent categories by modifying existing data in a controlled way. In the example dataset, we could flip the images with illnesses, or add noise to copies of the images in such a way that the illness remains visible.
3.	Using appropriate metrics. In the example dataset, if we had a model that always made negative predictions, it would achieve a precision of 98%. There are other metrics such as precision, recall, and F-score that describe the accuracy of the model better when using an imbalanced dataset.

**COMBAT THE CURSE OF DIMENSIONALITY?**
- Manual Feature Selection
- Principal Component Analysis (PCA)
- Multidimensional Scaling
- Locally linear embedding

##Model Selection and Design

**NLP中用CNN和RNN哪个更快，更好，为什么？**

当需要长记忆-RNN 如问答语义提取
需要提取更多特征-CNN 需要提取更多信息 比如name entity提取
有哪些activation function，CNN应该用哪个？为什么？
Activation: linear-> nonlinear (for universal approximation)
If no activation, then is just multiplication(linear)
Need to be Differentiable: calculate the derivative
Sigmoid(0, 1)/ tanh(-1, 1): vanishing gradient
ReLU for hidden layers: solves vanishing gradient
Leaky ReLU for hidden layers : solves bad Neurals(never updates)
Softmax for output: probability for different classes – sum to 1
Linear for output: for regression

**HOW TO CHOOSE THE NUMBER OF LAYERS OF NN?**

当且仅当数据非线性分离时才需要隐藏层。 多个隐藏层可以用于拟合非线性函数。对于一般简单的数据集，一两层隐藏层通常就足够了。
没有隐藏层：仅能够表示线性可分函数或决策
隐藏层数=1：可以拟合任何“包含从一个有限空间到另一个有限空间的连续映射”的函数
隐藏层数=2：搭配适当的激活函数可以表示任意精度的任意决策边界，并且可以拟合任何精度的任何平滑映射
隐藏层数>2：多出来的隐藏层可以学习复杂的描述（某种自动特征工程）
层数越深，理论上拟合函数的能力增强，效果按理说会更好，但是实际上更深的层数可能会带来过拟合的问题，同时也会增加训练难度，使模型难以收敛。
怎么样选择NN每层的neuron数目？
太多-overfit 太少underfit。

隐藏神经元的数量应在输入层的大小和输出层的大小之间。
隐藏神经元的数量应为输入层大小的2/3加上输出层大小的2/3。
隐藏神经元的数量考虑引入Batch Normalization, Dropout, 正则化等降低过拟合的方法。应小于输入层大小的两倍。
各个hidden layer 相同数目就行，或者递减，低层次提取更多的特征。

**BATCH NORM**

**REGULARIZATION**
- **L1 Lasso**
shrink some coefficients to zero, performing variable selection.
- **L2 Ridge**
shrinks all the coefficient by the same proportions but eliminates none

**DROPOUT**

##Deep Learning Training
**HYPER PARAMETER TUNING: HOW**
- Grid search hyper parameters with XGboost, 
```
{"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }
```

- Monitor the learning curve, better not learn too quick

**MODEL ARCHITECTURE TUNING: HOW**
1.	Tune the #hidden neuron in the network to prevent underfit and overfit
2.	layers higher the easier to overfit but less tendency to underfit
3.	Batch norm and dropout to prevent overfit and help with convergence

**BATCH SIZE INFLUENCE**

SGD中的batch size， Batch size determines how many examples you look at before making a weight update.
The lower it is, the noisier the training signal is going to be(Here one-data point at a time hence the gradient is aggressive (noisy gradients) hence there is going to be lot of oscillations. So there is a chance that your oscillations can make the algorithm not reach a local minimum. (diverge).),
the higher it is, the longer it will take to compute the gradient for each step. (too high can prevent convergence)
Use grid search for batch size and lr or other optimizer like ADAM.

**VANISHING GRADIENT/ EXPLODING GRADIENT/ DEAD NEURONS**

**UNDERFIT/ OVERFIT, BIAS/ VARIANCE**
- The bias error is an error from erroneous assumptions in the learning algorithm. 
High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).
- The variance is an error from sensitivity to small fluctuations in the training set. 
High variance can cause an algorithm to model the random noise in the training data, 
rather than the intended outputs (overfitting).

**HOW TO PREVENT OVERFIT?**
- Simpler model: less parameters
- Dropout/ Regularization
- Cross validation
![cv](images/cv.png)

**CONFUSION MATRIX**

n=165 | Predicted No | Predicted Yes
--- | --- | ---
Actual No | 50 | 10
Actual Yes | 5 | 100

- False Positive = 10
- False Negative = 5
- True Positive = 100
- True Negative = 50
- Accuracy = 150/165
- Recall = 100/ 150
- Precision = 100/110 (打中多少)
- F1-Score = 2 * (precision * recall) / (precision + recall)
##Ensemble

**BAGGING**

e.g. bootstrap. random sample with replacement.，然后在每组样本上训练出来的模型取平均。
Bagging是降低方差，prevent overfit

**BOOSTING**

e.g. Adaboost. 根据当前模型损失函数负梯度信息来训练新加入的弱分类器，
然后将训练好的弱分类器以累加的形式结合到现有的模型中 prevent underfit

![Boosting](images/boosting.png)

**WHY ENSEMBLE WORKS?**

An ensemble is the combination of multiple models to create a single prediction.
 The key idea for making better predictions is that the models should make 
 different errors. 
 That way the errors of one model will be compensated by the right guesses of 
 the other models and thus the score of the ensemble will be higher.

##System Design

**RECOMMENDER SYSTEM**

1.	Content based filter
2.	Collaborative filtering
