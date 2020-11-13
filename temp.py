import pandas as pd
import numpy as np
from collections import Counter
#读取数据

data_load_training = pd.read_csv("./data_download/x_train.csv",index_col=0)
data_load_training_label = pd.read_csv("./data_download/y_train_label.csv",index_col=0)

data_load_testing = pd.read_csv("./data_download/x_test.csv",index_col=0)
data_load_testing_label = pd.read_csv("./data_download/y_test_label.csv",index_col=0)

#合并
data_load_training["target"] =  data_load_training_label.values
data_load_testing["target"]  =  data_load_testing_label.values

print(Counter(data_load_training["target"]))
print(Counter(data_load_testing["target"]))