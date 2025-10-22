import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math

def entropy(y):
    counter = Counter(y)
    total = len(y)
    return -sum((count/total) * math.log2(count/total) for count in counter.values())


def info_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values = data[feature].unique()
    weighted_entropy = 0

    for v in values:
        subset = data[data[feature] == v]
        weighted_entropy += (len(subset)/len(data)) * entropy(subset[target])
    return total_entropy - weighted_entropy

def id3(data, features, target):
    # If all examples have the same label, return it
    if len(set(data[target])) == 1:
        return list(data[target])[0]

    # If no features left, return majority class
    if len(features) == 0:
        return Counter(data[target]).most_common(1)[0][0]

    # Choose best feature
    gains = [info_gain(data, f, target) for f in features]
    best_feature = features[np.argmax(gains)]

    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        subtree = id3(subset, remaining_features, target)
        tree[best_feature][value] = subtree

    return tree

def predict(tree, sample):
    if not isinstance(tree, dict):  # Leaf node
        return tree
    root = next(iter(tree))
    value = sample.get(root)
    if value in tree[root]:
        return predict(tree[root][value], sample)
    else:
        return None

def plot_tree(tree, depth=0, indent="  "):
    if not isinstance(tree, dict):
        print(indent * depth + f"--> {tree}")
        return
    for key, value in tree.items():
        print(indent * depth + str(key))
        for k in value:
            print(indent * (depth+1) + f"[{k}]")
            plot_tree(value[k], depth+2, indent)

if __name__ == "__main__":
    # Example dataset (Play Tennis)
    data = pd.read_csv("D:\\bharathcode\\deeplearning\\tennis.csv")
    target = 'PlayTennis'
    features = list(data.columns[:-1])

    # Build Decision Tree
    decision_tree = id3(data, features, target)

    print("\nDecision Tree:")
    plot_tree(decision_tree)

    # Classify a new sample
    new_sample = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
    prediction = predict(decision_tree, new_sample)
    print("\nNew Sample:", new_sample)
    print("Prediction:", prediction)
