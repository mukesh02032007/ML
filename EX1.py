import csv
with open("job_data.csv") as file:
    data = list(csv.reader(file))
# Separate header and training data
header = data[0]
training_data = data[1:]
# Initialize S and G
S = ['Ø'] * (len(header) - 1)
G = ['?'] * (len(header) - 1)
print("Initial S:", S)
print("Initial G:", G)
print("-" * 40)
# Process each example
for row in training_data:
    attributes = row[:-1]
    label = row[-1]
    if label == "Yes": # Positive example
        for i in range(len(S)):
            if S[i] == 'Ø':
                S[i] = attributes[i]
            elif S[i] != attributes[i]:
                S[i] = '?'
            else: # Negative example
                for i in range(len(G)):
                    if G[i] == '?' and S[i] != attributes[i]:
                        G[i] = S[i]
    print("Example:", row)
    print("S =", S)
    print("G =", G)
    print("-" * 40)
print("Final S:", S)
print("Final G:", G)