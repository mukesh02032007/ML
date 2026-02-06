import numpy as np
import math
import csv


# ---------- Read Dataset ----------
def read_data(dataset):
    with open(dataset, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        headers = next(datareader)
        metadata = headers
        data = [row for row in datareader]

    return metadata, np.array(data)


# ---------- Node Class ----------
class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ""


# ---------- Entropy ----------
def entropy(S):
    values, counts = np.unique(S, return_counts=True)
    ent = 0

    for c in counts:
        p = c / float(len(S))
        ent -= p * math.log(p, 2)

    return ent


# ---------- Subtables ----------
def subtables(data, col):
    tables = {}
    values = np.unique(data[:, col])

    for v in values:
        tables[v] = data[data[:, col] == v]

    return values, tables


# ---------- Information Gain ----------
def information_gain(data, col):
    total_entropy = entropy(data[:, -1])
    values, tables = subtables(data, col)

    weighted_entropy = 0
    for v in values:
        subset = tables[v]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[:, -1])

    return total_entropy - weighted_entropy


# ---------- Create Tree ----------
def create_node(data, metadata):

    # If all outputs same → Leaf
    if len(np.unique(data[:, -1])) == 1:
        leaf = Node("")
        leaf.answer = data[0, -1]
        return leaf

    # If no attributes left
    if len(metadata) == 1:
        leaf = Node("")
        leaf.answer = np.unique(data[:, -1])[0]
        return leaf

    # Compute gains
    gains = [information_gain(data, i) for i in range(len(metadata) - 1)]
    best_attr = np.argmax(gains)

    node = Node(metadata[best_attr])

    values, tables = subtables(data, best_attr)
    new_metadata = np.delete(metadata, best_attr)

    for v in values:
        child_data = np.delete(tables[v], best_attr, axis=1)
        child = create_node(child_data, new_metadata)
        node.children.append((v, child))

    return node


# ---------- Print Tree ----------
def print_tree(node, level=0):

    if node.answer != "":
        print(" " * level + "→ " + node.answer)
        return

    print(" " * level + "[" + node.attribute + "]")

    for value, child in node.children:
        print(" " * (level + 2) + str(value))
        print_tree(child, level + 4)


# ---------- Main ----------
metadata, data = read_data(
    r"D:\ML\weather_data.csv"
)

tree = create_node(data, metadata)

print_tree(tree)
