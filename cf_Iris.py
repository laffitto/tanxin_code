import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import load_iris


iris = load_iris()
y = iris['target']
X = iris['data']

#分为训练集和测试集
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)


kf = KFold(n_splits=5, shuffle=True, random_state=1)

#使用xgboost在训练集上做交叉验证
xgb_model_list = []
xgb_mse_list = []
for train_index, test_index in kf.split(train_X):
    xgb_model = xgb.XGBClassifier().fit(train_X[train_index], train_y[train_index])
    xgb_model_list.append(xgb_model)
    predictions = xgb_model.predict(train_X[test_index])
    actuals = train_y[test_index]
    #print(confusion_matrix(actuals, predictions))
    mse = mean_squared_error(actuals, predictions)
    xgb_mse_list.append(mse)

print ('xgb_mse_list:{}'.format(xgb_mse_list))
print ('xgb mse均值为:{}'.format(np.mean(xgb_mse_list)))


#使用随机森林在训练集上做交叉验证
rf_model_list = []
rf_mse_list = []
for train_index, test_index in kf.split(train_X):
    rf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=10)
    rf_model_list.append(rf)
    rf.fit(train_X[train_index], train_y[train_index])
    predictions = rf.predict(train_X[test_index])
    actuals = train_y[test_index]
    mse = mean_squared_error(actuals, predictions)
    rf_mse_list.append(mse)

print ('rf_mse_list:{}'.format(rf_mse_list))
print ('rf mse均值为:{}'.format(np.mean(rf_mse_list)))


#模型评估和选择
if np.mean(rf_mse_list) <= np.mean(xgb_mse_list):
    min_mse = min(rf_mse_list)
    ind = rf_mse_list.index(mse)
    best_estimator = rf_model_list[ind]
    print('best estimator is random forest {}, mse is {}'.format(ind,min_mse))
else:
    min_mse = min(xgb_mse_list)
    ind = xgb_mse_list.index(min(xgb_mse_list))
    best_estimator = xgb_model_list[ind]
    print('best estimator is xgb {}, mse is {}'.format(ind, min_mse))


#使用最好的模型和参数预测测试集，估计模型在实际使用时的判别能力
pred = best_estimator.predict(test_X)
mse = mean_squared_error(pred, test_y)
print ('test data mse is:{}'.format(mse))
print(confusion_matrix(test_y, pred))


