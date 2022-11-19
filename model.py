import pandas as pd
import numpy as np
import pickle
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('heart1.csv')
y = data["target"]
x = data.drop(columns=["target"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
fmodel = RandomForestClassifier(max_depth=5)
fmodel.fit(X_train, y_train)
pickle.dump(fmodel,open('Hridayai1.pkl','wb'))

# def heart_disease(sample):
#     data = pd.read_csv('heart1.csv')

#     y = data["target"]
#     x = data.drop(columns=["target"], axis=1)

#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

#     fmodel = KNeighborsClassifier(n_neighbors=5)
#     fmodel.fit(X_train, y_train)

#     out1 = pd.DataFrame(sample,columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs','restecg', 'thalach', 'exang', 'oldpeak','slope', 'ca', 'thal'])

#     y_pred = fmodel.predict(out1)
#     result = int(y_pred)
#     if(result==1):
#         return "RISK OF HEART DISEASE"
#     else:
#         return "RISK OF HEART DISEASE"