import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder, normalize
import pickle
import joblib

df = pd.read_csv("train.csv")
#print(df.isnull().any())

#print(df.info())

x = df[["roam_ic_mou","roam_og_mou","loc_ic_t2m_mou","std_og_mou"]]
y = df["Churned"]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,random_state = 42)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
#print(y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)
#print(acc*100)

joblib.dump(classifier, 'model.pkl')