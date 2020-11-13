import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import joblib
import tkinter
#导入自己的api.py,里面共有两个方法api.datachange和api.datachange2，用于特征工程
from api import data_utils as data_utils

#设置
model_m_name = "./train_model/logistc_train_model.m" #产生的模型名及路径
submission_name = "./submission/logistic_submisson.csv"


def modeltrain(xdata,ydata):
    #调用sklearn逻辑回归api
    #model = LogisticRegression(max_iter=100,random_state=10)
    model = KNeighborsClassifier(n_neighbors = 10, metric = 'manhattan')
    #标准化
    sc = StandardScaler()
    xdata = sc.fit_transform(xdata)
    #切分训练集
    training_features,testing_features,training_target,testing_target = train_test_split(xdata,ydata,test_size=0.3,random_state=25)
    #fit
    model = model.fit(training_features,training_target)
    #预测
    predict_target = model.predict(testing_features)
    #算准确率
    global acc
    acc = metrics.accuracy_score(testing_target,predict_target)
    #算auc
    global auc
    auc = metrics.roc_auc_score(testing_target,predict_target)
    #验证集上的auc值
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
    data_load = method.datachange(data_load).values
    #预测值的输出，并转化为df，并加上列名
    Survived = pd.DataFrame(model.predict(data_load))
    Survived.columns = ["Survived"]
    #df横向连接，输出为csv，不要标签
    pd.concat([PassengerId,Survived],axis = 1).to_csv(submission_name,index = 0)
    return
    
    
def main():
    #读取数据
    data_load = pd.read_csv("./data_download/train.csv")
    #调用自己的特征工程api
    global method
    method = data_utils.data_utils_method()
    #
    data_load = method.datachange(data_load)
    xdata,ydata = method.datachange2(data_load)
    model = modeltrain(xdata,ydata)
    modelout(model)
    print("模型已处理完毕")
    return


if __name__ == "__main__":
    main()