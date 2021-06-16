import time
import numpy as np
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_boston
import matplotlib
matplotlib.use('TkAgg')  # 'TkAgg' can show GUI in imshow()
# matplotlib.use('Agg')  # 'Agg' will not show GUI
from matplotlib import pyplot as plt
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin' # alter the graphviz bin path here
# plot_DT = False
plot_DT = True # comment this line if graphviz is not installed

# ================基于XGBoost原生接口的分类=============
def exp1():
    iris = load_iris()
    X,y = iris.data,iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565)

    # 算法参数
    params = {
        'booster': 'gbtree',
        # 'objective': 'multi:softprob',
        'objective': 'multi:softmax',
        'num_class': 3,
        'gamma': 0.1,
        'max_depth': 6,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'silent': 0,
        'eta': 0.1,
        'seed': 1000,
        'nthread': 4,
    }

    plst = list(params.items())
    dtrain = xgb.DMatrix(X_train, y_train) # 生成数据集格式
    num_rounds = 500
    # num_rounds = 1
    # num_rounds = 2
    model = xgb.train(plst, dtrain, num_rounds) # xgboost模型训练

    # 对测试集进行预测
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)

    # 显示重要特征
    plot_importance(model)
    plt.show()

    # 计算准确率
    accuracy = accuracy_score(y_test,y_pred)
    print("accuarcy: %.2f%%" % (accuracy*100.0))

    # 显示XGBoost学到的第几颗树
    if plot_DT is True:
        plot_tree(model, num_trees=0) # plot the decison tree for the 0-th tree, specify num_trees = n to show n-th tree
        plot_tree(model, num_trees=5, rankdir = 'LR') # 5-th tree, layout: from left to right

def val():
    W_T1 = [0.1245, -0.0636, -0.0665]
    W_T2 = [0.1128, -0.0613, -0.0679]
    W_Tree = W_T1
    pred_T1 = softmax(W_Tree)

    for i in range(3):
        W_Tree[i] += W_T2[i]
    pred_T2 = softmax(W_Tree)

    a = 1

def softmax(W):
    sumexp = 0
    for i in range(len(W)):
        sumexp += np.exp(W[i])
    soft_W = np.exp(W) / sumexp
    return soft_W

def main():
    exp1()
    # val()

if __name__ == "__main__":
    t_start = time.time()
    main()
    t_end = time.time()
    t_all = t_end - t_start
    print('Exp_XGBoost.py: whole time: {:.2f} min'.format(t_all / 60.))