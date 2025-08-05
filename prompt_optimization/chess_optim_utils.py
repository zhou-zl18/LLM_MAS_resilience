import re
import numpy as np
from collections import defaultdict
import os
import json

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

def parse_answer(content:str, previous_pos=None):
    content = content.lower()
    if previous_pos is not None:
        content = content.replace(previous_pos, '')
    
    pattern = r"[a-h][1-8]"
    
    # 如果有一行只包含a1格式，则为答案
    for line in content.split('\n'):
        tmp = line.strip()
        if len(tmp)==2:
            matches = re.findall(pattern, tmp)
            if len(matches) == 1:
                return matches[0].lower()
        
    pos = content.rfind("final answer")
    if pos != -1:
        """there exist final answer"""
        item = content.split("final answer")[-1].strip()
        matches = re.findall(pattern, item)
        if len(matches) == 1:
            return matches[0].lower()
        elif len(matches) > 1:
            """set the last one to answer"""
            # print([content])
            # print("*"*100)
            return matches[-1].lower()
        else:
            assert False, f"chess parse failed 1: \"{content}\""
    else:
        matches = re.findall(pattern, content)
        if len(matches) == 0:
            assert False, f"chess parse failed 2: \"{content}\""
        else:
            return matches[-1]
        


def evaluate_single_response(problem, response): # problem = {'input':xxx, 'target':xxx}
    task = problem['input']
    target = problem['target']
    game, previous_pos = task.rsplit(' ', 1)
    try:
        ans = parse_answer(response, previous_pos)
    except:
        ans = None
    output = ans in target
    return output

def calculate_acc(problems, task2messages, round_id=2):
    results = []
    for k, all_round_responses in task2messages.items():
        task = problems[k]['input']
        target = problems[k]['target']
        game, previous_pos = task.rsplit(' ', 1)

        ans_list = []
        round_responses = all_round_responses[round_id]
        for content in round_responses:
            try:
                ans = parse_answer(content[0], previous_pos)
            except:
                ans = None
            ans_list.append(ans)
        most_freq_ans = most_frequent_element(ans_list)
        results.append(most_freq_ans in target)
    acc = np.sum(results)/len(results)
    return acc

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