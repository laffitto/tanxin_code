import time
import xgboost as xgb
from xgboost import plot_importance,plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_boston
import matplotlib
matplotlib.use('TkAgg')  # 'TkAgg' can show GUI in imshow()
# matplotlib.use('Agg')  # 'Agg' will not show GUI
from matplotlib import pyplot as plt
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.37/bin' # alter the graphviz bin path here
plot_DT = False # comment this line if graphviz is not installed

# ================基于XGBoost原生接口的分类=============
def exp1():
    # 加载样本数据集
    iris = load_iris()
    # print(iris.DESCR)
    X,y = iris.data,iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565) # 数据集分割

    # 算法参数
    params = {
        'booster': 'gbtree',
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
    model = xgb.train(plst, dtrain, num_rounds) # xgboost模型训练

    # 对测试集进行预测
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)

    # 计算准确率
    accuracy = accuracy_score(y_test,y_pred)
    print("accuarcy: %.2f%%" % (accuracy*100.0))

    # 显示XGBoost学到的第几颗树
    if plot_DT is True:
        plot_tree(model, num_trees=0) # plot the decison tree for the 0-th tree, specify num_trees = n to show n-th tree
        plot_tree(model, num_trees=5, rankdir = 'LR') # 5-th tree, layout: from left to right

    # 显示重要特征
    plot_importance(model)
    # plt.show()

# ================基于XGBoost原生接口的回归=============
def exp2():
    # 加载数据集
    boston = load_boston()
    # print(boston.DESCR)
    X,y = boston.data,boston.target

    # XGBoost训练过程
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    params = {
        'booster': 'gbtree',
        'objective': 'reg:gamma',
        'gamma': 0.1,
        'max_depth': 5,
        'lambda': 3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'silent': 1,
        'eta': 0.1,
        'seed': 1000,
        'nthread': 4,
    }

    dtrain = xgb.DMatrix(X_train, y_train)
    num_rounds = 300
    plst = list(params.items())
    model = xgb.train(plst, dtrain, num_rounds)

    # 对测试集进行预测
    dtest = xgb.DMatrix(X_test)
    ans = model.predict(dtest)

    # 显示XGBoost学到的第几颗树
    if plot_DT is True:
        plot_tree(model, num_trees=0) # plot the decison tree for the 0-th tree, specify num_trees = n to show n-th tree
        plot_tree(model, num_trees=10, rankdir = 'LR') # 5-th tree, layout: from left to right

    # 显示重要特征
    plot_importance(model)
    # plt.show()

# ==============基于Scikit-learn接口的分类================
def exp3():
    # 加载样本数据集
    iris = load_iris()
    X,y = iris.data,iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565) # 数据集分割

    # 训练模型
    model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='multi:softmax')
    model.fit(X_train, y_train)

    # 对测试集进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test,y_pred)
    print("accuarcy: %.2f%%" % (accuracy*100.0))

    # 显示重要特征
    plot_importance(model)
    # plt.show()

    if plot_DT is True:
        plot_tree(model, num_trees=0, rankdir='LR')
        fig = plt.gcf()
        # fig.set_size_inches(150, 100)
        fig.savefig('tree.png')

# ================基于Scikit-learn接口的回归================
def exp4():
    boston = load_boston()
    X,y = boston.data,boston.target

    # XGBoost训练过程
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='reg:gamma')
    model.fit(X_train, y_train)

    # 对测试集进行预测
    ans = model.predict(X_test)

    # 显示XGBoost学到的第几颗树
    if plot_DT is True:
        plot_tree(model, num_trees=0) # plot the decison tree for the 0-th tree, specify num_trees = n to show n-th tree
        plot_tree(model, num_trees=2, rankdir = 'LR') # 5-th tree, layout: from left to right

    # 显示重要特征
    plot_importance(model)
    plt.show()

def main():
    exp1()
    # exp2()
    # exp3()
    # exp4()

if __name__ == "__main__":
    t_start = time.time()
    main()
    t_end = time.time()
    t_all = t_end - t_start
    print('Exp_XGBoost.py: whole time: {:.2f} min'.format(t_all / 60.))