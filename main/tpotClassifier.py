#coding=utf-8
#nohup python /root/titanic_python/tpotClassifier.py >log.out 2>&1 &
import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
#导入自己的api2.py,里面共有两个方法datachange和datachange2，用于特征工程
#api2处理原数据较少，删除缺失值，自动特征工程
from api import data_utils

#设置
#model_m_name = "./train_model/tpot_train_model.m" #产生的模型名及路径
model_m_name = "./train_model/tpot_train_model.py" #产生的模型名及路径
submission_name = "./submission/tpot_submisson.csv" #输出的预测文件名及路径


def modeltrain(xdata,ydata):
    #调用sklearn逻辑回归api
    model = TPOTClassifier(generations=1000,population_size=100,verbosity=2)
    #切分训练集
    training_features,testing_features,training_target,testing_target = train_test_split(xdata,ydata,test_size=0.3)
    #fit
    model = model.fit(training_features,training_target)
    #预测
    predict_traget = model.predict(testing_features)
    #算准确率
    acc = metrics.accuracy_score(testing_target,predict_traget)
    #打印准确率
    print("################################")
    print("the accuracy of test is :")
    print(acc)
    print("################################")
    #保存模型
    #joblib.dump(model, model_m_name)
    model.export(model_m_name)
    return model

def modelout(model):
    data_load = pd.read_csv("./data_download/test.csv")
    PassengerId = pd.DataFrame(data_load["PassengerId"])
    xdata,ydata = api.datachange(data_load)
    #预测值的输出，并转化为df，并加上列名
    Survived = pd.DataFrame(model.predict(xdata))
    Survived.columns = ["Survived"]
    #df横向连接，输出为csv，不要标签
    pd.concat([PassengerId,Survived],axis = 1).to_csv(submission_name,index = 0)
    return
    
    
def main():
    #读取数据
    data_load = pd.read_csv("./data_download/train.csv") 
    global api   
    api = data_utils.data_utils_method()
    xdata,ydata = api.datachange(data_load)
    model = modeltrain(xdata,ydata)
    modelout(model)
    print("Finished")
    return


if __name__ == "__main__":
    main()
