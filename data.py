import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder, normalize


df = pd.read_csv("churntrain.csv")
#print(df.isnull().any())

columns = ["roam_ic_mou", "roam_og_mou"]
encoder = OrdinalEncoder()
df[columns] = encoder.fit_transform(df[columns])
df1 = pd.DataFrame()
df1["roam_ic_mou"] = df["roam_ic_mou"]
df1["roam_og_mou"] = df["roam_og_mou"]
df1["loc_ic_t2m_mou"] = df["loc_ic_t2m_mou"]
df1["std_og_mou"] = df["std_og_mou"]
df1["Churned"] = df["Churned"]

df1.to_csv(r'C:\Users\ASUS\Desktop\Chrun\train.csv', index=False)