##nohup python /root/lutang_yiyao_python/tensor.py >./logs/log.out 2>&1 &
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from keras.wrappers import scikit_learn
#导入自己的api.py,里面共有两个方法datachange和datachange2，用于特征工程
from api import data_utils
import time 
time_now =  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 


#设置
model_path = "./train_model/tensor_model/" #产生的模型名及路径
submission_path = "./submission/tensor_submission/" #输出的预测文件名及路径

def modeltrain(xdata,ydata):
    #分测试集
    training_features,testing_features,training_target,testing_target = train_test_split(xdata,ydata)
    #分验证集
    training_features,validation_features,training_target,validation_target = train_test_split(training_features,training_target)
    def create_model():
        #tensorflow2.0的神经网络
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128,activation='relu',input_shape=(56,)),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        #keras的compile方法，定义损失函数、优化器和指标等
        model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=["acc"],
                ) #metrics输出正确率，它是一个列表
        #fit 带验证集
        return model
    #sklearn的自动调参,epoch和batch_size
    model = scikit_learn.KerasClassifier(build_fn=create_model, verbose=2)
    
    # 设置参数候选值
    batch_size = [8,16]
    epochs = [10,50]

    # 创建GridSearchCV，并训练
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(training_features, training_target)

    # 打印结果
    print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, std, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, std, param))
        return model

    
def main():
    #读取数据


    data_load_training = pd.read_csv("./data_download/X.csv",index_col=0)
    data_load_training_label = pd.read_csv("./data_download/y.csv",index_col=0)

    #合并
    data_load_training["target"] =  data_load_training_label.values


    
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
