
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


import pandas as pd
df = pd.read_csv("/Users/xuhong/Desktop/machine learning/hw2/Treasury Squeeze test - DS1.csv" )
X = df.iloc[:,2:11]

y = df.iloc[:, 11]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
tree = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
tree.fit(X_train_std, y_train)
y_pred = tree.predict(X_test)
print(accuracy_score(y_pred,y_test))



