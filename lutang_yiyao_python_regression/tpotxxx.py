##nohup python /root/lutang_yiyao_python/tpotxxx.py >./logs/log.out 2>&1 &
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from tpot import TPOTClassifier
import joblib
#导入自己的api.py,里面共有两个方法datachange和datachange2，用于特征工程
from api import data_utils

import time 
time_now =  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 


#设置
model_path = "./train_model/tpot_model/" #产生的模型名及路径
submission_path = "./submission/tpot_submission/" #输出的预测文件名及路径

def modeltrain(xdata,ydata):
    #分测试集
    training_features,testing_features,training_target,testing_target = train_test_split(xdata,ydata)
    #tpot
    model = TPOTClassifier(generations=20,population_size=50,verbosity=3,scoring='roc_auc')
    #fit
    model.fit(training_features,training_target)
    #分好0-1
    predict_target_classes = model.predict(testing_features)
    #没分0-1 输出的是概率
    predict_target = model.predict_proba(testing_features)
    #print(predict_target)
    #print(predict_target_classes)
    #算准确率
    acc = metrics.accuracy_score(testing_target,predict_target_classes)
    #算auc
    auc = metrics.roc_auc_score(testing_target,predict_target[:,1])
    #算matric
    confusion_matrix = metrics.confusion_matrix(testing_target,predict_target_classes)
    #输出的名字
    global output_name
    output_name = time_now + " auc " + str(round(auc,4)) + " acc " + str(round(acc,4)) + " "

    #打印准确率
    print("*************************")
    print(r"the acc of test_data is :")
    print(acc)
    print(r"the auc of test_data is :")
    print(auc)
    print(r"the confusion_matrix of test_data is :")
    print(confusion_matrix)
    #保存模型
    model.export(model_path + output_name + '.py')
    return model
'''
def modelout(model):
    data_load = pd.read_csv("./data_download/test.csv")
    PassengerId = pd.DataFrame(data_load["PassengerId"])
    xdata,ydata = api.datachange(data_load)
    #预测值的输出，并转化为df，并加上列名
    Survived = pd.DataFrame((model.predict(xdata) > 0.5).astype("int32"))
    Survived.columns = ["Survived"]
    #df横向连接，输出为csv，不要标签
    pd.concat([PassengerId,Survived],axis = 1)\
        .to_csv(submission_path + output_name + ".csv",index = 0)
    return
'''    
    
def main():
    #读取数据
    data_load_training = pd.read_csv("./data_download/X.csv",index_col=0)
    data_load_training_label = pd.read_csv("./data_download/y.csv",index_col=0)
    #合并
    data_load_training["target"] =  data_load_training_label.values
    #调用api
    global api
    api = data_utils.data_utils_method()
    #训练集x y
    xdata,ydata = api.datachange(data_load_training)
    #训练模型
    model = modeltrain(xdata,ydata)
    #modelout(model)
    print(r"Model Finished")
    print("*************************")
    return


if __name__ == "__main__":
    main()
# %%
