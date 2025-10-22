import pandas as pd

def candidate_elimination(filename):
    # Load dataset
    data = pd.read_csv("D:\\bharathcode\\deeplearning\\data.csv")
    concepts = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values

    # Step 1: Initialize S and G
    S = [['0'] * len(concepts[0])]
    G = [['?'] * len(concepts[0])]

    print("\nInitial S:", S)
    print("Initial G:", G)

    # Step 2: Process each training example
    for i, h in enumerate(concepts):
        if target[i].lower() == "yes":  # positive example
            # Remove from G inconsistent hypotheses
            G = [g for g in G if all(g[j] in ['?', h[j]] for j in range(len(h)))]

            # Generalize S
            for j in range(len(S[0])):
                if S[0][j] == '0':
                    S[0][j] = h[j]
                elif S[0][j] != h[j]:
                    S[0][j] = '?'

        else:  # negative example
            # Remove from S inconsistent hypotheses
            if all(S[0][j] in ['?', h[j]] for j in range(len(h))):
                # Specialize G
                new_G = []
                for j in range(len(S[0])):
                    if S[0][j] == '?':
                        new_hypothesis = S[0].copy()
                        new_hypothesis[j] = h[j] + "_not"  # mark specialization
                        new_G.append(new_hypothesis)
                G.extend(new_G)

        print(f"\nAfter example {i+1} ({h}, {target[i]}):")
        print("S:", S)
        print("G:", G)

    return S, G


if __name__ == "__main__":
    S_final, G_final = candidate_elimination("training_data.csv")
    print("\nFinal Specific Boundary (S):", S_final)
    print("Final General Boundary (G):", G_final)
