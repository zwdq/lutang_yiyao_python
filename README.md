# titanic_python
训练kaggle
下载下来，项目目录里python xgb.py即可
其他模型同理 各py文件分别是不同模型
api.py里是简易特征工程,自己改
输出在submission里

tpot.py是自动建模模型，github上的项目，基因迭代不同管道，输出最好的模型及参数

tensor是谷歌的tensorflow框架，windows上要vc++，核心是keras类，通过tf.keras.models.Sequential构造网络，输入可以是张量，目前示例输入还是矩阵，是简单情况，张量是高纬矩阵，方法同理，看官网的tensor类
