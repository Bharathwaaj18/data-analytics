import pandas as pd

def find_s_algorithm(filename):
    # Load CSV
    data = pd.read_csv("D:\\bharathcode\\deeplearning\\data.csv")

    # Separate features and target
    attributes = data.iloc[:, :-1].values   # all columns except last
    target = data.iloc[:, -1].values        # last column
    
    # Step 1: Initialize most specific hypothesis
    hypothesis = ['0'] * (attributes.shape[1])
    
    # Step 2: Iterate through training examples
    for i, val in enumerate(attributes):
        if target[i].lower() == "yes":  # positive example
            for j in range(len(hypothesis)):
                if hypothesis[j] == '0':
                    hypothesis[j] = val[j]
                elif hypothesis[j] != val[j]:
                    hypothesis[j] = '?'
    
    return hypothesis


if __name__ == "__main__":
    final_hypothesis = find_s_algorithm("training_data.csv")
    print("Final Hypothesis:", final_hypothesis)
