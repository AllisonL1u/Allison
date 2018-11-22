############################################################
################ chaos like shit :( ########################
############################################################






# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn import cross_validation
from sklearn import metrics 
from sklearn import model_selection

#import re
from xgboost import XGBClassifier
import xgboost as xgb

train_df = pd.read_csv('/Users/allisonliu/project/kaggle/titanic/train.csv')
test_df = pd.read_csv('/Users/allisonliu/project/kaggle/titanic/test.csv')
combine = [train_df, test_df]


######## A. Featrue Engineering #######

############# 1.na of data analysis #########
######### numerical feature values & categorical features
train_df.info()
print('_'*40)
test_df.info()

train_df.describe()
train_df.describe(include=['O'])

test_df.describe()
test_df.describe(include=['O'])


######################### 2.how to fill na data ############

######################### 3.features -> numerical ################
######################### one-hot coding: title 6/embarked 3

######################### 4.suitable model chosen for data & confirm hyper-parameters

############# 

############# 

######## B. Model Evaluation & Parameters Modification (Cross Validation) #######


######## C. Model Ensemble (Stacking) #######



############### !!!!! Ticket: Extract Information ################

from collections import OrderedDict

def set_ticket(s):
    if s.isdigit():
        return 'D'  # means digit
    else:
        s = s.split(' ')[0]   # get prefix
        s = s.replace('.', '')   # remove '.'
        s = s.replace('/', '')   # remove '/'
        s = "".join(OrderedDict.fromkeys(s))    # unique()
        s = s[0]    # first char
        if s in 'AWFL':
            s = 'R'    # 'R' means rare
        return s[0]



for dataset in combine:
    dataset["Ticket"] = dataset["Ticket"].apply(set_ticket)
    dataset["Ticket"] = dataset["Ticket"].map({'D': 1, 'S': 2, 'P': 3, 'R': 4, 'C': 5}).astype('uint8')

# train_df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean().sort_values(by='Ticket', ascending=True)


##################### Title: New feature extracted from names ###############

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


########################### Age: continous -> category (cut)#####################

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['Fare'].fillna(train_df['Fare'].dropna().median(), inplace=True)


### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(train_df, test_df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    df = pd.concat([train_df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']], test_df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]], axis=0)
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    #df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # y即目标年龄
    y = known_age[:, 0]
    # X即特征属性值
    X = known_age[:, 1:]
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    train_df['Age'] = df['Age'][:891]
    test_df['Age'] = df['Age'][891:]
    #train_df[ (df.Age.isnull()), 'Age' ] = predictedAges[:train_df[train_df.Age.isnull()].shape[0]]
    #test_df[ (df.Age.isnull()), 'Age' ] = predictedAges[train_df[train_df.Age.isnull()].shape[0]:]
    return train_df, test_df, rfr


train_df, test_df, rfr = set_missing_ages(train_df, test_df)
# predictedAges = rfr.predict(unknown_age[:, 1::])
# # 用得到的预测结果填补原缺失数据
# df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges




train_df['AgeBand'] = pd.cut(train_df['Age'], 4)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 20.315, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 20.315) & (dataset['Age'] <= 40.21), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 40.21) & (dataset['Age'] <= 60.105), 'Age'] = 2
    dataset.loc[ dataset['Age'] > 60.105, 'Age'] = 3

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

# ################## Age: fillna referring age and class(median)##############
# guess_ages = np.zeros((2,3))
# for dataset in combine:
#     for i in range(0, 2):
#         for j in range(0, 3):
#             guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
#             age_guess = guess_df.median()
#             guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
#     for i in range(0, 2):
#         for j in range(0, 3):
#             dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]


# dataset['Age'] = dataset['Age'].astype(int)


########################### Age*Class: new feature ###########################

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

####################### IsAlone: new feature(Sibsp + Parch) #####################

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1


train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

##################### Embarked: 1.fill na value with the most frequent ################
##################### Embarked: 2.categorical -> numerical ############################
freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

##################### Fare: 1.fill na value with the median value #####################
##################### Fare: 2. continous -> category (qcut) #####################

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['Fare'].fillna(train_df['Fare'].dropna().median(), inplace=True)

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

########## Sex ################
for dataset in combine:
    dataset["Sex"] = dataset["Sex"].map({'male':0, 'female': 1}).astype('uint8')



combine = [train_df, test_df]
############ cabin ################

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = 1
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = 0
    df["Cabin"] = df["Cabin"].astype('uint8')
    return df

train_df = set_Cabin_type(train_df)
test_df = set_Cabin_type(test_df)



def setDummyVariable(dummies, df):
	for i in dummies:
		print(i)
		dummy = pd.get_dummies(df[i], prefix = i)
		df = pd.concat([df, dummy], axis=1)
	return df

train_df = setDummyVariable(['Title', 'Embarked', 'Age'], train_df)
train_df.drop(['Name', 'Embarked', 'Embarked_Q', 'Title', 'Age'], axis=1, inplace=True)

train_df.drop(['PassengerId'], axis=1, inplace=True)

test_df = setDummyVariable(['Title', 'Embarked', 'Age'], test_df)
test_df.drop(['Name', 'Embarked', 'Embarked_Q', 'Title', 'Age'], axis=1, inplace=True)
#dummies_Title = pd.get_dummies(train_df['Title'], prefix= 'Title')

train_df.columns
test_df.columns

############ Pearson Correlation Heatmap ############ 

colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_df.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
plt.xticks(rotation=90)    # 将字体进行旋转  
plt.yticks(rotation=360)  
plt.show()






X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

85.629999999999995
xgb

Accuracy : 0.8462
AUC Score (Train): 0.909256

def modelfit(alg, dtrain, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
        metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    xgb.plot_importance(alg)
    plt.show()

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
target = 'retain_status'
#IDcol = 'PassengerId'
predictors = [x for x in train_df.columns if x not in [target]]
xgb1 = XGBClassifier(learning_rate = 0.1, n_estimators = 183, max_depth = 3,
min_child_weight = 3, reg_alpha=0.001, gamma = 0.1, subsample = 0.6, colsample_bytree = 0.8,
 objective = 'binary:logistic', nthread = 1, scale_pos_weight = 1, seed = 100)
modelfit(xgb1, train_df, predictors, target)

param_test1 = {
  'reg_alpha':[i/10 for i in range(1,10)]
}
gsearch1 = model_selection.GridSearchCV( estimator = XGBClassifier(learning_rate = 0.1, n_estimators = 200, max_depth = 3,
min_child_weight = 3, reg_alpha=0.01, gamma = 0.1, subsample = 0.6, colsample_bytree = 0.8, objective = 'binary:logistic', nthread = 1, scale_pos_weight = 1, seed = 100), param_grid = param_test1, scoring = 'roc_auc', n_jobs = 1, iid = False, cv = 5)
gsearch1.fit(train_df[predictors],train_df[target])
gsearch1.cv_results_
gsearch1.best_params_, gsearch1.best_score_


({'max_depth': 3, 'min_child_weight': 3}, 0.87380532370325648)
({'gamma': 0.1}, 0.87476187175018683)
({'n_estimators': 183}, 0.87600248715361695)
({'colsample_bytree': 0.8, 'subsample': 0.6}, 0.87820238448209553)
({'seed': 100}, 0.87862662619531628)
({'reg_alpha': 0.001}, 0.87870567757871532)




xgbc = XGBClassifier(learning_rate = 0.1, n_estimators = 183, max_depth = 3,
min_child_weight = 3, reg_alpha=0.001, gamma = 0.1, subsample = 0.6, colsample_bytree = 0.8,
 objective = 'binary:logistic', nthread = 1, scale_pos_weight = 1, seed = 100)
#xgbc = XGBClassifier()
xgbc.fit(X_train, Y_train)
Y_pred = xgbc.predict(X_test)
acc_xgbc = round(xgbc.score(X_train, Y_train) * 100, 2)
acc_xgbc

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log

# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

# KNN

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

# Gaussian Naive Bayes

gnb = GaussianNB()
gnb.fit(X_train, Y_train)
Y_pred = gnb.predict(X_test)
acc_gaussian = round(gnb.score(X_train, Y_train) * 100, 2)
acc_gaussian


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd

# Decision Tree

dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
Y_pred = dt.predict(X_test)
acc_decision_tree = round(dt.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# Random Forest

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, Y_train)
Y_pred = rfc.predict(X_test)
rfc.score(X_train, Y_train)
acc_random_forest = round(rfc.score(X_train, Y_train) * 100, 2)
acc_random_forest

# Model evaluation

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'XGBoost'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree, acc_xgbc]})
models.sort_values(by='Score', ascending=False)

#########################  Local Cross Validation & BadCases ###################################

def localCrossValidation(model, df):
    #简单看看打分情况
    rf = model
    all_data = df.filter(regex='Survived|Age_.*|Title_.*|IsAlone|Fare|Cabin|Ticket|Embarked_.*|Sex|Pclass|Age*Class')
    X = all_data.as_matrix()[:,1:]
    y = all_data.as_matrix()[:,0]
    print(cross_validation.cross_val_score(rf, X, y, cv=5))

svc = SVC(C = 0.5, gamma = 0.1, kernel = 'rbf')
logreg = LogisticRegression(C = 1, class_weight = None, max_iter = 100, penalty = 'l2')

localCrossValidation(logreg, train_df)
sum(map(float,a.split()))/5



def badCases(train_df, myModel):
    # 分割数据，按照 训练数据:cv数据 = 7:3的比例
    split_train, split_cv = cross_validation.train_test_split(train_df, test_size=0.3, random_state=0)
    train_df = split_train.filter(regex='Survived|Age_.*|Title_.*|IsAlone|Fare|Cabin|Ticket|Embarked_.*|Sex|Pclass|Age*Class')
    myModel.fit(train_df.as_matrix()[:,1:], train_df.as_matrix()[:,0])
    # 对cross validation数据进行预测
    cv_df = split_cv.filter(regex='Survived|Age_.*|Title_.*|IsAlone|Fare|Cabin|Ticket|Embarked_.*|Sex|Pclass|Age*Class')
    predictions = myModel.predict(cv_df.as_matrix()[:,1:])
    origin_data_train = pd.read_csv("/Users/allisonliu/project/kaggle/titanic/train.csv")
    bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
    return bad_cases

# 生成模型
myModel = svc
name = str(myModel).split('(')[0]
badCase = badCases(train_df, myModel)
badCase.to_csv('/Users/allisonliu/project/kaggle/titanic/badCase_'+name+'.csv', index=False)



############################  Grid Search 确定模型参数  ###################################
# 1.Random Forest #############################################################
# max_features -> sqrt(features)
# n_estimators 森林中树木的数目
param_grid = {
    'n_estimators': [50, 100, 200, 300, 500], 
    'max_features': [2, 3, 4, 5, 6]
}
rfc = RandomForestClassifier()
model = model_selection.GridSearchCV(estimator=rfc, param_grid=param_grid, n_jobs=1, cv=5, verbose=20, scoring='accuracy')


model.fit(X_train, Y_train)
print('The score of the best model is: ')
model.best_score_
print('The parameters of the best model are: ')
model.best_params_


# >>> model.best_score_ 
# 0.81818181818181823
# >>> model.best_params_ 
# {'max_features': 3, 'n_estimators': 100}

# 2.SVM ####################################################################################
#- kernel（核函数linear、RBF）
#- C（ C是惩罚系数，即对误差的宽容度。c越高，说明越不能容忍出现误差,容易过拟合。C越小，容易欠拟合。C过大或过小，泛化能力变差）
#- gamma（ gamma是选择RBF函数作为kernel后，该函数自带的一个参数。隐含地决定了数据映射到新的特征空间后的分布，gamma越大，支持向量越少，gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。）

param_grid = {
    'C' : [0.1, 0.5, 1.0, 2.0, 4.0], 
    'gamma' : [0.1 ,0.2, 0.4, 0.57, 0.8, 1.6, 3.2], 
    'kernel' : ['linear', 'rbf', 'poly', 'sigmoid']
}
svc = SVC()
model = model_selection.GridSearchCV(estimator=svc, param_grid=param_grid, n_jobs=1, cv=6, verbose=20, scoring='accuracy')


model.fit(X_train, Y_train)
print('The score of the best model is: ')
model.best_score_
print('The parameters of the best model are: ')
model.best_params_


#>>> model.best_score_
#0.81818181818181823
#>>> model.best_params_
#{'C': 0.57, 'gamma': 0.575, 'kernel': 'rbf'}

# 3.KNearestNeighbor ###############################################################
# leaf_size:build tree
# n_neighbors & weights

param_grid = {
	'leaf_size' : [7, 10, 20, 25, 30, 35, 40], 
	'n_neighbors' : [1, 3, 5, 7, 9, 11],
	'weights' : ['uniform', 'distance']
}
knn = KNeighborsClassifier(algorithm = 'auto', metric = 'minkowski')
model = model_selection.GridSearchCV(estimator=knn, param_grid=param_grid, n_jobs=1, cv=7, verbose=20, scoring='accuracy')


model.fit(X_train, Y_train)
print('The score of the best model is: ')
model.best_score_
print('The parameters of the best model are: ')
model.best_params_



# >>> model.best_score_
# 0.81481481481481477
# >>> model.best_params_
# {'leaf_size': 20, 'n_neighbors': 3, 'weights': 'uniform'}

# 4. Logistic Regression #############################################################
# class_weight 分类模型中各种类型的权重
# penalty 惩罚项
# C 正则化系数λ的倒数 
# max_iter 最大迭代次数
param_grid  = {
	'C' : [0.01, 0.1, 1, 10, 100], 
	'class_weight' : [None, 'balanced'], 
    'max_iter' : [100,300,500], 
    'penalty' : ['l1','l2']
}
logreg = LogisticRegression()
model = model_selection.GridSearchCV(estimator=logreg, param_grid=param_grid, n_jobs=1, cv=6, verbose=20, scoring='accuracy')

model.fit(X_train, Y_train)
print('The score of the best model is: ')
model.best_score_
print('The parameters of the best model are: ')
model.best_params_


# >>> model.best_score_
# 0.82267115600448937
# >>> model.best_params_
# {'C': 1, 'class_weight': None, 'max_iter': 100, 'penalty': 'l1'}

# 5.Guassion Naive Bayes #############################################################
# 
param_grid  = {}
gnb = GaussianNB()
model = model_selection.GridSearchCV(estimator=gnb, param_grid=param_grid, n_jobs=1, cv=10, verbose=20, scoring='accuracy')


model.fit(X_train, Y_train)
print('The score of the best model is: ')
model.best_score_
print('The parameters of the best model are: ')
model.best_params_

# >>> model.best_score_
# 0.77890011223344557


##########################################################################################


def getSubmission(Y_pred):
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
    submission.to_csv('/Users/allisonliu/project/kaggle/titanic/submission.csv', index=False)


getSubmission(Y_pred)


class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        kf = model_selection.KFold(n_splits=self.n_folds, shuffle=True, random_state=2018)
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], self.n_folds))
            for j, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(1)
        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred, self.stacker


rfc = RandomForestClassifier(max_features = 3, n_estimators = 100)
svc = SVC(C = 0.5, gamma = 0.1, kernel = 'rbf')
knn = KNeighborsClassifier(leaf_size = 20, n_neighbors = 3, weights = 'uniform', algorithm = 'auto', metric = 'minkowski')
logreg = LogisticRegression(C = 1, class_weight = None, max_iter = 100, penalty = 'l2')
xgbc = XGBClassifier(learning_rate = 0.1, n_estimators = 183, max_depth = 3,
min_child_weight = 3, reg_alpha=0.001, gamma = 0.1, subsample = 0.6, colsample_bytree = 0.8,
 objective = 'binary:logistic', nthread = 1, scale_pos_weight = 1, seed = 100)

myEnsemble = Ensemble(5, xgbc, [xgbc, svc, rfc, logreg, knn])
y_pred_stacking, stacker = myEnsemble.fit_predict(X_train, Y_train, X_test)
getSubmission(y_pred_stacking)
localCrossValidation(stacker, train_df)

sum(map(float,a.split()))/5

############################### Stacking ###################################
