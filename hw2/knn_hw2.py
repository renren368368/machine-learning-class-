from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("/Users/xuhong/Desktop/machine learning/hw2/Treasury Squeeze test - DS1.csv" )
X = df.iloc[:,2:11]

y = df.iloc[:, 11]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
k_range = range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test,y_pred))
print(scores)

plt.plot(k_range, scores, color="r", linestyle="-", marker="^", linewidth=1)
plt.xlabel("random_state")
plt.ylabel("scores")
plt.show()
