# coding: utf-8

import pickle
import matplotlib.pyplot as plot
from tools.feature_format import featureFormat
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from tools.feature_format import targetFeatureSplit
from sklearn.cross_validation import train_test_split 

enron_data = pickle.load(open('./final_project/final_project_dataset.pkl','rb'))

# ======== 工具方法 ===========
def get_features_data(input_data, features_list):
    """
    调用featureFormat获取数据
    """
    features_data = featureFormat(input_data, features_list)
    return features_data

def get_all_features(enron_data):
    """
    获取所有的features
    """
    keys = list(enron_data.keys())
    features_dict = enron_data[keys[0]]
    features = features_dict.keys()
    return list(features)

def check_nan(feature_key):
    results = []
    for name ,feature in enron_data.items():
        if feature[feature_key] == 'NaN':
            results.append(name)
    return results

def get_and_fix_nan(data,key):
    """
    从字典中取值，如果是NaN，就修正为0.0
    """
    value = data[key]
    if value == 'NaN':
        value = 0.
    return value

def get_total(key1, key2):
    """
    获取加和计算之后的列表
    """
    total = {}
    for name ,feature in enron_data.items():
        value1 = get_and_fix_nan(feature, key1)
        value2 = get_and_fix_nan(feature, key2)
        total[name]=(value1+value2)
    return total

def deep_copy(src_list):
    """
    将集合的元素全部复制到另一个集合，用来代替=赋值
    src_list: 源集合或者元组
    """
    return_list = [src_list[index] for index in range(len(src_list))]
    return return_list

def get_min_max(data_list):
    """
    获取数据列表中的最大值最小值
    """
    return min(data_list), max(data_list)

# ========= 探索数据 ===========
enron_keys = list(enron_data.keys())
keys_as_set = set(enron_keys)

# 两个不是姓名 THE TRAVEL AGENCY IN THE PARK, TOTAL，从数据集中删除
if 'THE TRAVEL AGENCY IN THE PARK' in enron_data.keys():
    enron_data.pop('THE TRAVEL AGENCY IN THE PARK')
if 'TOTAL' in enron_data.keys():
    enron_data.pop('TOTAL',0)
print("数据个数 %s " %(str(len(enron_data))))

# feature列表
all_features = get_all_features(enron_data)
print("所有的feature个数:",len(all_features))
print(all_features)

# POI name:
poi_names = []
non_poi_names = []
nan_poi_names = []
for name, feature in enron_data.items():
    if feature['poi'] == 1:
        poi_names.append(name)
    elif feature['poi']==0:
        non_poi_names.append(name)
    else :
        nan_poi_names.append(name)
        
print("所有的poi共有%d个;非poi个数%d;poi值是nan的%d" % (len(poi_names),len(non_poi_names),len(nan_poi_names)))

# 主要特征中包含NAN数据的
# Salary
salary_nans = check_nan('salary')
salary_nans
print("共有%d个salary为nan" % len(salary_nans))

# total_payments
total_payments_nans = check_nan("total_payments")
print("共有%d个total_payments为nan" % len(total_payments_nans))

# long_term_incentive
long_term_incentive_nans = check_nan('long_term_incentive')
print("共有%d个long_term_incentive为nan" % (len(long_term_incentive_nans)))

# total_stock_value
total_stock_value_nans = check_nan('total_stock_value')
print("%d个total_stock_value为nan" % len(total_payments_nans))

#### 主要特征都是NaN的
many_nans = []
for name in total_payments_nans:
    if name in salary_nans and name in long_term_incentive_nans and name in total_stock_value_nans and name in total_payments_nans:
        many_nans.append(name)

for name in many_nans:
    print(name)
    print(enron_data[name])
# 看到LOCKHART EUGENE E对应的所有features都是NaN, 应该是异常点,从原始数据中删除
enron_data.pop('LOCKHART EUGENE E',0)

original_data_path = 'final_project/my_dataset.pkl'
pickle.dump(enron_data,open(original_data_path,'wb'))

# ======== 探究变量之间的关系 ========
def plot_variance(features_name_list):
    """
    绘制两个变量的散点图
    """
    features_data = featureFormat(enron_data, features_name_list)
    
    for point in features_data:
        plot.scatter(point[0], point[1])
    
    plot.xlabel(features_name_list[0])
    plot.ylabel(features_name_list[1])
    plot.show()
    return features_data

# salary, bonus
features_name_list = ['salary', 'bonus']
#return_data = plot_variance(features_name_list)

# #### 从散点图可以看到有2个明显的异常点，bonus和salary都远高于其他人，是否是poi？


# 找到他们的name
salary_map_name = {}
name_map_poi = {}
for name ,feature in enron_data.items():
    if feature['salary'] == 'NaN':
        continue
    salary_map_name[feature['salary']] = name
    name_map_poi[name] = feature['poi']
    
top_2_salary = sorted(list(salary_map_name.keys()), reverse=True)[:2]
top_2_salary_names = [salary_map_name[salary] for salary in top_2_salary]
for n in top_2_salary_names:
    print("\'%s\' is poi ? %d" % (n, name_map_poi[n]))


# #### 根据计算结果，两个人都是poi

# #### emails.
# #### 增加新的属性ratio_email_with_poi. 根据邮件中与POI交互邮件所占比来分析是否是poi。
# #### 注意NaN和0的处理.


email_with_poi = get_total('from_poi_to_this_person', 'from_this_person_to_poi')
total_email = get_total('to_messages', 'from_messages')

ratio_email_with_poi = {}
for name, email_count in email_with_poi.items():
    if total_email[name] == 0.:
        ratio_email_with_poi[name] = 0.
    else:
        ratio_email_with_poi[name] = email_count/total_email[name]

for name, features in enron_data.items():
    total = total_email[name]
    if total==0:
        features['ratio_to_poi']=0.
        features['ratio_from_poi']=0.
    else:
        features['ratio_to_poi']=features['from_this_person_to_poi']/total
        features['ratio_from_poi']=features['from_poi_to_this_person']/total

# 在enron_data中增加新的属性
for name, feature in enron_data.items():
    feature['ratio_email_with_poi'] = ratio_email_with_poi[name]
    
# 将上面的enrondata保存到新的pkl
enron_data_new_path = 'final_project/enron_data_new.pkl'
pickle.dump(enron_data, open(enron_data_new_path, 'wb'))
    
    
# 取出包含poi和ratio的列表,分析两个变量之间的关系
ratio_map_poi = {}
for name, feature in enron_data.items():
    is_poi = 0.
    if feature['poi'] == 'True' or feature['poi'] == True:
        is_poi = 1.0
    ratio_map_poi[feature['ratio_email_with_poi']] = is_poi
    
# 绘制两种分类的箱线图
"""
non_pois = [ratio for ratio in ratio_map_poi if ratio_map_poi[ratio]<=0]
pois = [ratio for ratio in ratio_map_poi if ratio_map_poi[ratio]>=1]
plot.boxplot(x=[non_pois,pois], labels=['non_poi', 'poi'])
plot.show()
"""

# #### 从箱线图的分位数上可以看出，poi的人的邮件比重明显高于非poi的。因此可以将邮件比重作为新的属性来分析是不是poi。

# ========= 训练模型 ===================
# 从features中移除email_address
enron_data = pickle.load(open(original_data_path,'rb'))
features = get_all_features(enron_data)
features_no_address = deep_copy(features)
features_no_address.remove('email_address')
features_no_address_new = deep_copy(features)
features_no_address_new.remove('email_address')
features_no_address_new.append('ratio_email_with_poi')

# 决策树分类
def prepare_data(input_data, features_list):
    """
    准备分类器需要的features,target数据
    """
    data_format = featureFormat(input_data, features_list)
    targets, features = targetFeatureSplit(data_format)
    features_train, features_test, target_train, target_test = train_test_split(features, targets, test_size = 0.3, random_state=42)
    
    from sklearn.cross_validation import KFold
    kf=KFold(len(targets),3)
    for train_indices, test_indices in kf:
        #make training and testing sets
        features_train= [features[ii] for ii in train_indices]
        features_test= [features[ii] for ii in test_indices]
        target_train=[targets[ii] for ii in train_indices]
        target_test=[targets[ii] for ii in test_indices]
    return features_train, features_test, target_train, target_test

count =0 

def tree_classifier(input_data, features_list) :
    """
    使用决策树
    """
    global count
    count+=1
    print("=============%d============" % count)
    features_train, features_test, target_train, target_test = prepare_data(input_data, features_list)
    
    clf = DecisionTreeClassifier()
    clf = clf.fit(features_train, target_train)
    pred = clf.predict(features_test)
    accu = accuracy(target_test, pred)
    print("准确率%f" % accu)
    prec = precision_score(target_test, pred)
    rec = recall_score(target_test, pred)
    f1 = f1_score(target_test, pred)
    print("精度%f" % prec)
    print("召回率%f" % rec)
    print("f1 %f" % f1)
    
    importance = clf.feature_importances_
    indices = list(numpy.argsort(importance))
    indices = reversed(indices)
    important_features = []
    for no, index in enumerate(indices):
        if importance[index]>0:
            print("No.%d--属性%s的权重%f" % (no, features_list[index+1], importance[index]))
            important_features.append(features_list[index+1])

    return important_features, clf
    
    
#  使用原始数据
"""
data_original = pickle.load(open(original_data_path, 'rb'))

features_no_address.remove('poi')
features_no_address.insert(0, 'poi')
important_features = tree_classifier(data_original, features_no_address)
print("重要变量",important_features)
important_features.insert(0, 'poi')
tree_classifier(data_original, important_features)

# 新数据：
data_new = pickle.load(open(enron_data_new_path, 'rb'))

features_no_address_new.remove('poi')
features_no_address_new.insert(0, 'poi')
important_features = tree_classifier(data_new, features_no_address_new)
print("重要变量",important_features)

#  增加新的变量后，准确率没有上升。所以还是采用原来的features
# ======== 选择参数 ===================
"""
# SelectKBest
data_original = pickle.load(open(original_data_path, 'rb'))
features_no_address = get_all_features(data_original)
features_no_address.remove('email_address')
features_no_address.remove('poi')
features_no_address.insert(0, 'poi')
features_no_address.remove('shared_receipt_with_poi')
features_train, features_test, targets_train, targets_test = prepare_data(data_original, features_no_address)


selector = SelectKBest(k=10)
selector = selector.fit(features_train, targets_train)
features_selection = selector.get_support(indices=True)
features_names_selected = [features_no_address[index] for index in features_selection]
print("被选择的feature: ", features_names_selected)
""""
# 使用上面选择的feature再次进行决策树分类
data_original = pickle.load(open(original_data_path, 'rb'))
features_selected = ['poi','deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred', 'expenses', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'shared_receipt_with_poi']
tree_classifier(data_original, features_selected)

# ============== 选择其他算法： ==============

# Naive_bayes
data_original = pickle.load(open(original_data_path, 'rb'))
#print(data_original)
print(important_features)
features_train, features_test, targets_train, targets_test = prepare_data(data_original, important_features)
nb = GaussianNB()
nb = nb.fit(features_train, targets_train)
pred = nb.predict(features_test)
accu = accuracy(targets_test, pred)
pre = precision_score(targets_test, pred)
recall = recall_score(targets_test, pred)

print("朴素贝叶斯模型准确率: %f" % accu)
print("精度: %f" % pre)
print("召回率: %f" % recall)

# SVC
data_original = pickle.load(open(original_data_path, 'rb'))
# 特征缩放
scale_features=['deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value']
for f in scale_features:
    feature_values=[]
    for name, features in data_original.items():
        value = features[f]
        if value == 'NaN':
            value = 0.
        feature_values.append(value)
    
    _min = min(feature_values)
    _max = max(feature_values)
    if _min == _max:
        continue
    for name, features in data_original.items():
        value = features[f]
        if value == 'NaN':
            value = 0.
        features[f] = (value-_min)/(_max-_min)
        
features_train, features_test, targets_train, targets_test = prepare_data(data_original, important_features)
clf = LinearSVC()
clf = clf.fit(features_train, targets_train)
pred = clf.predict(features_test)
accu = accuracy(targets_test, pred)
pre = precision_score(targets_test, pred)
recall = recall_score(targets_test, pred)
print("支持向量机模型准确率: %f" % accu)
print("精度: %f" % pre)
print("召回率: %f" % recall)

# #### 参数调整

important_features.insert(0, 'poi')
important_features = tree_classifier(data_original, important_features)
print("重要参数",important_features)
"""
# 使用GridSearchCV
features_selected = ['poi','ratio_to_poi','ratio_from_poi','total_payments','total_stock_value','long_term_incentive']
data_original = pickle.load(open(enron_data_new_path, 'rb'))
pickle.dump(data_original,open(original_data_path,'wb'))
feature_important, clf = tree_classifier(data_original , features_selected)

"""
features_train, features_test, targets_train, targets_test = prepare_data(data_original, features_selected)
params = {
         'min_samples_split' : range(5,80,5), 'splitter' : ('best', 'random')
          }
clf = GridSearchCV(DecisionTreeClassifier(), params)
clf = clf.fit(features_train, targets_train)

# 再次对该模型进行计算
pred = clf.predict(features_test)
accu = accuracy(targets_test, pred)
precision = precision_score(targets_test, pred)
recall = recall_score(targets_test, pred)
print("准确率%f" % accu)
print("精度%f" % precision)
print("召回率%f" % recall)
"""
my_classifier_path = 'final_project/my_classifier.pkl'
my_features_path = 'final_project/my_feature_list.pkl'
pickle.dump(clf,open(my_classifier_path, 'wb'))
pickle.dump(features_selected, open(my_features_path, 'wb'))

print("程序结束")