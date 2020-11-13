import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import joblib
import tkinter
import tkinter.messagebox 
#导入自己的api.py,里面共有两个方法datachange和datachange2，用于特征工程
from api import data_utils

#设置
model_m_name = "./train_model/xgb_train_model.m" #产生的模型名及路径
submission_name = "./submission/xgb_submisson.csv" #输出的预测文件名及路径


def modeltrain(xdata,ydata,n_numbers,lr):
    #调用sklearn逻辑回归api
    model = XGBClassifier(
        learning_rate=lr,       # 学习速率
        #reg_alpha=1,            # l1正则权重
        n_estimators=n_numbers, # 树的个数 --n棵树建立xgboost
        max_depth=6,            # 树的深度
        min_child_weight=2,
        nthread=1,
        subsample=0.85,          # 随机选择x%样本建立决策树，小了欠拟合，大了过拟合
        #colsample_bytree = 0.9, # x%特征建立决策树
        #scale_pos_weight=1,     # 解决样本个数不平衡的问题
        #gamma=0.1, 
        #random_state=20,        # 随机数
        #objective='binary:logistic',  # 损失函数 objective='multi:softmax' 'binary:logistic' reg:linear
        #wram_start=True,
        )
    #切分前标准化
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
    #取出测试集的x,ydata默认为空
    xdata,ydata = method.datachange(data_load)
    #预测值的输出，并转化为df，并加上列名
    Survived = pd.DataFrame(model.predict(xdata))
    Survived.columns = ["Survived"]
    #df横向连接，输出为csv，不要标签，输出在submission里，只有id和输出值两列
    pd.concat([PassengerId,Survived],axis = 1).to_csv(submission_name,index = 0)
    return
    
#主函数弄俩参数给modeltrain用
def main(n_numbers,lr):
    #读取数据
    data_load = pd.read_csv("./data_download/train.csv")
    #特征工程第一步,method是api文件夹里特征工程.py的类实例化，做个尝试，global是因为其他的def里没法直接用
    global method
    method = data_utils.data_utils_method()
    #取出训练集的x，y数组
    xdata,ydata = method.datachange(data_load)
    #输入modeltrain函数的参数，输出模型
    model = modeltrain(xdata,ydata,n_numbers,lr)
    modelout(model)
    print("模型已处理完毕")
    return


if __name__ == "__main__":
    window = tkinter.Tk()
    window.title("xgb建模调参")
    #画组件
    tkinter.Label(window, text="学习率：").grid(row=0)
    tkinter.Label(window, text="训练器个数：").grid(row=1)
    tkinter.Label(window, text="auc面积：").grid(row=2)
    tkinter.Label(window, text="acc准确率：").grid(row=3)
    e1 = tkinter.Entry(window)
    e2 = tkinter.Entry(window)
    e3 = tkinter.Entry(window)
    e4 = tkinter.Entry(window)
    e1.grid(row=0, column=1, padx=10, pady=5)
    e2.grid(row=1, column=1, padx=10, pady=5)
    e3.grid(row=2, column=1, padx=10, pady=5)
    e4.grid(row=3, column=1, padx=10, pady=5)
    e1.insert(0, "0.1")
    e2.insert(0, "100")
    e3.insert(0, "暂无 ")
    e4.insert(0, "暂无 ")
    
    #按钮的功能
    def button1():
        n_numbers = int(e2.get())
        lr = float(e1.get())
        #主函数入口
        main(n_numbers,lr)
        #tkinter.messagebox.showinfo("auc曲线结果是",auc,parent=window)
        e3.delete(0, "end")
        e3.insert(0, round(auc,4))
        e4.delete(0, "end")
        e4.insert(0, round(acc,4))
        #e1.delete(0, "end")
        #e2.delete(0, "end")

    tkinter.Button(window, text="开始建模", width=10, command=button1).grid(row=4, column=0, sticky="w", padx=10, pady=5)
    tkinter.Button(window, text="退出", width=10, command=window.quit).grid(row=4, column=1, sticky="e", padx=10, pady=5)
    window.mainloop()

