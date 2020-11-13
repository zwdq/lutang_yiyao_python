import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
import tkinter
#导入自己的api.py,里面共有两个方法api.datachange和api.datachange2，用于特征工程
from api import data_utils

#设置
model_m_name = "./train_model/logistc_train_model.m" #产生的模型名及路径
submission_name = "./submission/logistic_submisson.csv"


def modeltrain(xdata,ydata,max_iter=100,random_state=10):
    #调用sklearn逻辑回归api
    model = LogisticRegression(max_iter=max_iter,random_state=random_state)
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
    
    
def main(max_iter=100,random_state=10):
    #读取数据
    data_load = pd.read_csv("./data_download/train.csv")
    #调用自己的特征工程api
    global method
    method = data_utils.data_utils_method()
    #
    data_load = method.datachange(data_load)
    xdata,ydata = method.datachange2(data_load)
    model = modeltrain(xdata,ydata,max_iter,random_state)
    modelout(model)
    print("模型已处理完毕")
    return


if __name__ == "__main__":
    window = tkinter.Tk()
    window.title("logistic建模调参")
    #画组件
    tkinter.Label(window, text="迭代次数：").grid(row=0)
    tkinter.Label(window, text="随机数：").grid(row=1)
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
    e1.insert(0, "1000")
    e2.insert(0, "23")
    e3.insert(0, "暂无 ")
    e4.insert(0, "暂无 ")
    
    #按钮的功能
    def button1():
        max_iter = int(e1.get())
        random_state = int(e2.get())
        #主函数入口
        main(max_iter,random_state)
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