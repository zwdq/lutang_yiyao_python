##nohup python /root/lutang_yiyao_python/tensor.py >./logs/log.out 2>&1 &
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import joblib
#导入自己的api.py,里面共有两个方法datachange和datachange2，用于特征工程
from api import data_utils

import time 
time_now =  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 


#设置
model_path = "./train_model/xgb_model/" #产生的模型名及路径
submission_path = "./submission/xgb_submission/" #输出的预测文件名及路径

def modeltrain(xdata,ydata):
    #分测试集
    training_features,testing_features,training_target,testing_target = train_test_split(xdata,ydata)
    #xgb
    model = XGBClassifier(
        learning_rate=0.1,        # 学习速率
        #silent= 0,  # 为0打印运行信息；设置为1静默模式，不打印
        #reg_alpha=1,           # l1正则权重
        n_estimators=1000,      # 树的个数 --n棵树建立xgboost
        max_depth=6,            # 树的深度
        min_child_weight=2,
        nthread=1,
        subsample=0.85,          # 随机选择x%样本建立决策树，小了欠拟合，大了过拟合
        #colsample_bytree = 0.9, # x%特征建立决策树
        #scale_pos_weight=1,     # 解决样本个数不平衡的问题
        #gamma=0.1, 
        random_state=20,        # 随机数
        eval_metric='auc',
        #objective='binary:logistic',  # 损失函数 objective='multi:softmax' 'binary:logistic' reg:linear
        #wram_start=True,
        )

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
    joblib.dump(model,model_path + output_name + '.m')
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
