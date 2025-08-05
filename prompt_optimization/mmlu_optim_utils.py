import re
import numpy as np
from collections import defaultdict
import os
import json

def get_adj_matrix(adj_list):

    # Convert adj_list to adj_matrix (a numpy array)
    num_nodes = len(adj_list)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    node_ids = sorted([int(k) for k in adj_list.keys()])
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    for node_str, neighbors in adj_list.items():
        i = id_to_idx[int(node_str)]
        for neighbor in neighbors:
            j = id_to_idx[int(neighbor)]
            adj_matrix[i, j] = 1

    return adj_matrix

def calculate_resilience(accs: list): # accs: accuracy for error_rate = 0, 0.2, 0.4, 0.6, 0.8
    orig_acc = accs[0]
    if orig_acc == 0:
        return 0
    tmp = []
    for acc in accs:
        if acc < orig_acc:
            tmp.append(acc)
        else:
            tmp.append(orig_acc)
    resilience = (tmp[0] + 2 * np.sum(tmp[1:]) + 0)/ 10 / orig_acc
    return resilience

from collections import Counter
def most_frequent_element(lst):
    lst = [x for x in lst if x is not None]

    counter = Counter(lst)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None

        

def evaluate_single_response(response, answer):
    response = response.strip()
    response = response[-1].upper()
    output = response in answer
    return output



