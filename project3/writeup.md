
#### 一. 数据记录
* 数据点总数：144
* Poi个数：18个，非poi个数：126个。
* feature数量：21个。
* feature中有NaN值的:
    * 共有50个salary为nan
    * 共有21个total_payments为nan
    * 共有79个long_term_incentive为nan
    * 21个total_stock_value为nan
    * LOCKHART EUGENE E对应的所有features都是NaN


##### 异常点:
* 打印所有的name，发现两个不属于人名的点：THE TRAVEL AGENCY IN THE PARK, TOTAL
* 分析salary和bonus两个变量的散点图，发现两个点对应的salary和bonus都远高于其他点，打印姓名为SKILLING JEFFREY K和LAY KENNETH L。打印两个人的poi属性，发现都=1.将这两个点作为异常值，使用dict.pop()从数据集中删除
* salary-bonus散点图如下:

Figure.png![image.png](attachment:image.png)

#### 优化和选择特征

##### 增加新的变量 ratio_email_with_poi
* 根据邮件中与POI交互邮件所占比来分析是否是poi.比值越大，是poi的嫌疑越大.
* 计算ratio_email_with_poi并绘制与变量poi的箱线图.

Figure_1.png![image.png](attachment:image.png)

##### 特征选择


features.png![image.png](attachment:image.png)

* 如何选择features：
    * 对数据集进行决策树训练，从训练模型的变量权重中，提取出权重比较高的变量作为新的变量集合再次训练。如果最终准确率精确率都得到明显提升，则采用这个变量集合。
* 对原数据集合和全部变量采用SelectKBest选取最优变量，继续进行决策树模型训练，准确率和精确率没有明显上升。
* 结合上表，综合三个测量标准采用表格中第四行的features

##### 特征缩放：
* 对SVC模型进行特征缩放。
* 对选取的feature列表，使用min_max缩放器对数值进行缩放，使所有的feature都在0-1之间
    * 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value'
    * 上面这几个变量变化区间过大，因此对他们进行缩放

#### 选择算法和调整参数

##### 调整参数
* 就是对模型中的一些参数进行动态调整，以使模型达到最优。在本项目的解答中，通过设定参数'min_samples_split': [2, 3, 4, 5, 6, 7, 8],'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],'max_features': range(3,10)对决策树模型进行调整。

##### 选择其他算法：
* 朴素贝叶斯：准确率0.72,精确率0.2，召回率0.2
* 支持向量机: 准确率0.69, 精确率0, 召回率0

* 最终选择决策树模型，因为准确率和精确率较高。

#### 验证和评估
* 验证是从原始数据中分离出一部分作为测试数据，来评估模型的准确率。本项目解答中，在训练时通过sklearn包中的相关模块，将原始数据分割为训练数据和测试数据，再通过accuracy_score模块对最终的训练模型精确率进行评估; 二次训练时使用GridSearchCV对模型进行交叉验证。
* 未进行验证，结果会过拟合。所有数据都是训练数据，模型会尽量偏重每个数据的正确性，从而导致模型过拟合。

* 准确率：模型预测正确的/全部。
* 假设：预测结果是真--PT，预测结果是假--PF,实际是真--TT, 实际是假--TF
   * 精确率：TT&&PT/(TT&&PT+TF&&PT)
   * 召回率: TT&&PT/(TT&&PT+TT&&PF)
   

* precision=0.2表示是poi并且被预测为poi的人占所有被预测为poi的人的20%
* recall=0.16表示是poi并且被预测为poi的人占是poi的人的预测结果的16%

* tester.py中的startifiedshufflesplit:
* sklearn.cross_validation.startifiedshufflesplit:
    * 分离训练集和测试集。
    * 这个方法大致是，对数据集随机选取训练集和测试集，将他们代入模型计算准确率，并重复此过程，最终计算平均准确率。
