#!/usr/bin/python
# coding: utf-8

import sys
import pickle
import tools
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

all_features = tools.getKeys(data_dict.values()[0])

### Task 2: Remove outliers
features_list = ['poi','salary','bonus']

# 获取所有特征
features_list = []
for feature in all_features:
    for point in data_dict.values():
        value = point[feature]
        if value=='NaN':
            continue
        if type(value)==type(1) or type(value)==type(1.):
            if feature!='poi' and feature not in features_list:
                features_list.append(feature)
features_list.insert(0, 'poi')
print features_list

data_dict.pop('TOTAL',0)
featureDatas = featureFormat(data_dict, features_list)

def drawscatter():
    features = data_dict.values()

    for feature in featureDatas:
        salary = feature[0]
        bonus = feature[1]
        plot.scatter(salary, bonus)

    plot.xlabel('salary')
    plot.ylabel('bonus')
    plot.show()

# 发现异常值，salary和bonus远高于其他人
# 打印发现名字是TOTAL,从集合中移除
# 再次绘制散点图发现散点图右上角仍然有数据，即bonus和salary都高于其他人,SKILLING JEFFREY K，LAY KENNETH L。
# 这两个人是正常数据点，不做处理
#  移除 TOTAL异常值之后，使用朴素贝叶斯,决策树训练分类准确率为0.886,0.841,0.886

salary = [featureDatas[i][0] for i in range(len(featureDatas))]
salary.sort()
salary_max = salary[len(salary)-1]
for name , feature in data_dict.items():
    # if feature['salary'] == salary_max:
    #     print name, salary_max
    salary = feature['salary']
    bonus = feature['bonus']
    if  salary!='NaN' and bonus!='NaN' and salary>=1000000 and bonus>=5000000:
        print name

### Task 3: Create new feature(s)
# 创建新的变量，保存来往邮件中，与poi交互的邮件百分比。百分比越高，表示与poi联系越密切，越可能是poi
# 增加新变量之后，运行分类器预测结果，准确率分别为0.886,0.864,0.886
def checkNaN(point):
    if point == 'NaN':
        point=0
    return point

total_messages=[]
total_poi_messages = []
for name, features in data_dict.items():
    frommsg = features['from_messages']
    tomsg = features['to_messages']
    from_poi = features['from_poi_to_this_person']
    to_poi = features['from_this_person_to_poi']
    frommsg = checkNaN(frommsg)
    tomsg = checkNaN(tomsg)
    from_poi = checkNaN(from_poi)
    to_poi = checkNaN(to_poi)
    total = frommsg+tomsg
    total_poi = from_poi + to_poi
    ratio = 0
    if total != 0:
        ratio = float(total_poi)/total
    features['poi_total_ratio'] = ratio
    data_dict[name] = features
features_list.remove('from_messages')
features_list.remove('to_messages')
features_list.remove('from_poi_to_this_person')
features_list.remove('from_this_person_to_poi')
### Store to my_dataset for easy export below.
my_dataset = data_dict

# features_list = ['poi','salary','poi_total_ratio']
features_list.insert(len(features_list),'poi_total_ratio')

features_list = ['poi', 'total_payments', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'expenses', 'other', 'deferred_income', 'long_term_incentive']
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train,labels_train)
score = clf.score(features_test, labels_test)
print 'First time run , score is: ', score


# 使用决策树
from sklearn import tree
decisionTree = tree.DecisionTreeClassifier()
decisionTree.fit(features_train, labels_train)
print "Use decision tree : ", decisionTree.score(features_test, labels_test)
feature_importance = decisionTree.feature_importances_
print feature_importance
# 获取特征指数大于0的特征:
most_important = []
for index in range(len(feature_importance)):
    if feature_importance[index]>0.:
        most_important.append(features_list[1:][index])
print most_important

# 使用GridSearchCV进行参数调整。
from sklearn.model_selection import GridSearchCV
parameters = {'min_samples_split':range(2,50,2), 'random_state':range(0,40,5), 'splitter':('random','best') }
gridSearchCV = GridSearchCV(decisionTree, parameters)
gridSearchCV.fit(features_train, labels_train)
accuracy = gridSearchCV.score(features_test, labels_test)
print '参数调整后的预测：', accuracy
# 参数调整之后预测准确率为0.86
# 使用支持向量机
from sklearn.svm import SVC
svc = SVC()
svc.fit(features_train, labels_train)
print "Use svm: ", svc.score(features_test, labels_test)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
# 什么是验证:
# 验证是用于评估模型好坏的一个重要方法，
# 将数据集分为训练集和测试集是为了避免过拟合同时观察训练器准确率。
# 训练集用来建立训练模型，测试集用来评估模型的准确率。
# 通过验证测试集来确定训练集是否“过拟合”或者“欠拟合”。
dump_classifier_and_data(clf, my_dataset, features_list)
