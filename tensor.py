##nohup python /root/lutang_yiyao_python/tensor.py >./logs/log.out 2>&1 &
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
#导入自己的api.py,里面共有两个方法datachange和datachange2，用于特征工程
from api import data_utils
import time 
time_now =  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 


#设置
model_path = "./train_model/tensor_model/" #产生的模型名及路径
submission_path = "./submission/tensor_submission/" #输出的预测文件名及路径

def modeltrain(xdata,ydata,testing_features,testing_target):
    training_features=xdata
    training_target=ydata
    #切分训练集为训练集和测试集
    #再次切分训练集为训练集和验证集用于建模；这样，总共有三组样本：训练集，验证集和测试集
    #主要是因为现在还不会tensorflow的交叉验证
    #training_features,validation_features,training_target,validation_target = train_test_split(training_features,training_target)
    #tensorflow2.0的神经网络
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(input_shape=(56,)),
        tf.keras.layers.Dense(16,),
        tf.keras.layers.Dense(16,activation='relu'),
        tf.keras.layers.Dense(16),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    #keras的compile方法，定义损失函数、优化器和指标等
    model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=[tf.keras.metrics.AUC()],
             ) #metrics输出正确率，它是一个列表
    #fit
    model.fit(training_features,training_target,validation_data=(testing_features,testing_target),epochs=500,batch_size=8,verbose=2)
    #分好0-1
    predict_target_classes = (model.predict(testing_features) > 0.5).astype("int32")
    #没分0-1 输出的是概率
    predict_target = model.predict(testing_features)
    #算准确率
    acc = metrics.accuracy_score(testing_target,predict_target_classes)
    #算auc
    auc = metrics.roc_auc_score(testing_target,predict_target)
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
    model.save(model_path + output_name + '.h5')
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

    data_load_training = pd.read_csv("./data_download/x_train.csv",index_col=0)
    data_load_training_label = pd.read_csv("./data_download/y_train_label.csv",index_col=0)

    data_load_testing = pd.read_csv("./data_download/x_test.csv",index_col=0)
    data_load_testing_label = pd.read_csv("./data_download/y_test_label.csv",index_col=0)

    #合并
    data_load_training["target"] =  data_load_training_label.values
    data_load_testing["target"]  =  data_load_testing_label.values


    
    global api
    api = data_utils.data_utils_method()
    #训练集x y
    xdata,ydata = api.datachange(data_load_training)
    #测试集x y
    testing_features,testing_target = api.datachange(data_load_testing)
    #训练模型
    model = modeltrain(xdata,ydata,testing_features,testing_target)
    #modelout(model)
    print(r"Model Finished")
    print("*************************")
    return


if __name__ == "__main__":
    main()
# %%
