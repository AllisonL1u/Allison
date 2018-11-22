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

#from sklearn import cross_validation
from sklearn.model_selection import cross_validate
from sklearn import metrics 
from sklearn import model_selection

#import re
from xgboost import XGBClassifier
import xgboost as xgb


train_df = pd.read_csv('/Users/xmly/Downloads/feature.csv')



############ Pearson Correlation Heatmap ############ 
def showPearsonHeatmap(train_df):
	colormap = plt.cm.RdBu
	plt.figure(figsize=(14,12))
	plt.title('Pearson Correlation of Features', y=1.05, size=15)
	sns.heatmap(train_df.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
	plt.xticks(rotation=90)    # 将字体进行旋转  
	plt.yticks(rotation=360)  
	plt.show()




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
xgb1 = XGBClassifier(learning_rate = 0.1, n_estimators = 300, max_depth = 3,
min_child_weight = 3, reg_alpha=0.1, gamma = 0.1, subsample = 0.6, colsample_bytree = 0.8,
 objective = 'binary:logistic', nthread = 1, scale_pos_weight = 1, seed = 66)
modelfit(xgb1, train_df, predictors, target)


def gridSearch(alg, train_df, predictors, target):
    param_test1 = {
        'reg_alpha':[i/10 for i in range(5,15)]
        #,'min_child_weight':[i for i in range(1,4)]
        ,'learning_rate':[i/30 for i in range(1,5)]
    }
    gsearch1 = model_selection.GridSearchCV( estimator = alg, param_grid = param_test1, scoring = 'roc_auc', n_jobs = 1, iid = False, cv = 5)
    gsearch1.fit(train_df[predictors],train_df[target])
    return gsearch1



alg = XGBClassifier(learning_rate = 0.05, n_estimators = 200, max_depth = 2,
					min_child_weight = 1, reg_alpha=0.9, gamma = 0.1, 
					subsample = 0.6, colsample_bytree = 0.8, 
					objective = 'binary:logistic', nthread = 1, 
					scale_pos_weight = 1, seed = 100)
res = gridSearch(alg, train_df, predictors, target)
res.best_params_, res.best_score_


X_train = train_df.drop("retain_status", axis=1)
Y_train = train_df["retain_status"]



def localCrossValidation(model, df):
    #简单看看打分情况
    rf = model
    all_data = df#.filter(regex='Survived|Age_.*|Title_.*|IsAlone|Fare|Cabin|Ticket|Embarked_.*|Sex|Pclass|Age*Class')
    X = all_data.as_matrix()[:,1:]
    y = all_data.as_matrix()[:,0]
    return cross_validate(rf, X, y, cv=5)

svc = SVC(C = 0.5, gamma = 0.1, kernel = 'rbf')
logreg = LogisticRegression(C = 1, class_weight = None, max_iter = 100, penalty = 'l2')


ret = localCrossValidation(xgbc, train_df)
ret['train_score'].mean()
ret['test_score'].mean()




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










