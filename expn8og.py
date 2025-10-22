import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv("tennis.csv")

X = data.drop("PlayTennis", axis=1)
y = data["PlayTennis"]

X = pd.get_dummies(X)

clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X, y)

plt.figure(figsize=(15, 10))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
plt.show()
