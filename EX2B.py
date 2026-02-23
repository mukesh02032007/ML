import csv
import math
def read_csv(file):
    with open(file) as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [row for row in reader]
    return header, data
def entropy(col):
    counts = {}
    for val in col:
        counts[val] = counts.get(val, 0) + 1
    ent = 0
    for count in counts.values():
        p = count / len(col)
        ent -= p * math.log2(p)
    return ent
def info_gain(data, col_idx, target_idx):
    target_col = [row[target_idx] for row in data]
    total_entropy = entropy(target_col)
    vals = set(row[col_idx] for row in data)
    weighted = 0
    for v in vals:
        subset = [row[target_idx] for row in data if row[col_idx] == v]
        weighted += (len(subset) / len(data)) * entropy(subset)
    return total_entropy - weighted
def build_tree(data, attributes, target_idx):
    target_col = [row[target_idx] for row in data]
    if target_col.count(target_col[0]) == len(target_col):
        return target_col[0]
    if len(attributes) == 0:
        return max(set(target_col), key=target_col.count)
    gains = [info_gain(data, i, target_idx) for i in range(len(attributes))]
    best = gains.index(max(gains))
    tree = {attributes[best]: {}}
    vals = set(row[best] for row in data)
    for v in vals:
        subset = [[row[i] for i in range(len(row)) if i != best]
                  for row in data if row[best] == v]
        new_attrs = [attributes[i] for i in range(len(attributes)) if i != best]
        new_target_idx = target_idx - 1 if target_idx > best else target_idx
        tree[attributes[best]][v] = build_tree(subset, new_attrs, new_target_idx)
    return tree
def print_tree(tree, indent=""):
    if isinstance(tree, dict):
        for key, val in tree.items():
            print(indent + str(key))
            for v, subtree in val.items():
                print(indent + "  ->", v)
                print_tree(subtree, indent + "    ")
    else:
        print(indent + "Answer:", tree)
filename = "weather_data.csv"
attributes, data = read_csv(filename)
target_idx = len(attributes) - 1
tree_id3 = build_tree(data, attributes[:-1], target_idx)
print("--- ID3 Decision Tree ---")
print_tree(tree_id3)
