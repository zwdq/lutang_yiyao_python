import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFwe, f_classif
#导入自己的api.py,里面共有两个方法datachange和datachange2，用于特征工程
from api import api

#设置
model_m_name = "./train_model/random_train_model.m" #产生的模型名及路径
submission_name = "./submission/random_submisson.csv" #输出的预测文件名及路径


def modeltrain(xdata,ydata):
    #调用sklearn逻辑回归api
    '''
    model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=1, min_samples_leaf=19, min_samples_split=11)),
    SelectFwe(score_func=f_classif, alpha=0.004),
    RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.25, min_samples_leaf=1, min_samples_split=18, n_estimators=100)
)
    '''
    #下面这个是kaggle kernel上跑出来的，效果看起来不错，提交后acc0.787
    model = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.4, min_samples_leaf=4, min_samples_split=16, n_estimators=100)
    #切分训练集
    training_features,testing_features,training_target,testing_target = train_test_split(xdata,ydata,test_size=0.25,random_state=None)
    #fit
    model = model.fit(training_features,training_target)
    #预测
    predict_target = model.predict(testing_features)
    #算准确率
    acc = metrics.accuracy_score(testing_target,predict_target)
    #算auc
    auc = metrics.roc_auc_score(testing_target,predict_target)#验证集上的auc值
    #打印auc
    print("auc值是:")
    print(auc)
    #打印准确率
    print("准确率是:")
    print(acc)
    #保存模型
    joblib.dump(model, model_m_name)
    return model

def modelout(model):
    data_load = pd.read_csv("./data_download/test.csv")
    PassengerId = pd.DataFrame(data_load["PassengerId"])
    data_load = api.datachange(data_load).values
    #预测值的输出，并转化为df，并加上列名
    Survived = pd.DataFrame(model.predict(data_load))
    Survived.columns = ["Survived"]
    #df横向连接，输出为csv，不要标签
    pd.concat([PassengerId,Survived],axis = 1).to_csv(submission_name,index = 0)
    return
    
    
def main():
    #读取数据
    data_load = pd.read_csv("./data_download/train.csv")
    data_load = api.datachange(data_load)
    xdata,ydata = api.datachange2(data_load)
    model = modeltrain(xdata,ydata)
    modelout(model)
    print("模型已处理完毕")
    return


if __name__ == "__main__":
    main()