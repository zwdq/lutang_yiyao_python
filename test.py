import pandas as pd
from collections import Counter

data_load_training = pd.read_csv("./data_download/X.csv",index_col=0)
data_load_training_label = pd.read_csv("./data_download/y.csv",index_col=0)

#合并
data_load_training["target"] =  data_load_training_label.values
print(Counter(data_load_training["target"]))