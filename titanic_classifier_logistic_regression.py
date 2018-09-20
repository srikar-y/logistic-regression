import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df1 = pd.read_csv('H:/learning-area/titanic-dataset/train.csv')
df2 = pd.read_csv('H:/learning-area/titanic-dataset/test.csv')
df1.head()
df1.describe()
df2.head()
df2.describe()

train = df1.copy()
test = df2.copy()
data = pd.concat([train, test], ignore_index=True, sort=False)
data.info()

data['Age'].fillna(data['Age'].median(), inplace=True)
data.isnull().sum()
data['Fare'].fillna(data['Fare'].mean(), inplace=True)
data.isnull().sum()
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.isnull().sum()
delete_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)
data.info()

dummy = pd.get_dummies(data['Sex'])
data_new = pd.concat([data, dummy], axis=1)
dum = pd.get_dummies(data_new['Embarked'])
data_1 = pd.concat([data_new, dum], axis=1)
del_col = ['Sex', 'Embarked']
data_1.drop(del_col, axis=1, inplace=True)

train = data_1[:891]
test = data_1[891:]
target = train['Survived']
train.drop('Survived', axis=1, inplace=True)

from sklearn import model_selection
train_x, val_x, train_y, val_y = model_selection.train_test_split(train, target, random_state=0)
train_x.info()
val_x.info()

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model = LogisticRegression()
model.fit(train_x, train_y)
val_y_pred = model.predict(val_x)
model.score(val_x, val_y)

from sklearn.metrics import classification_report
print(classification_report(val_y, val_y_pred))

