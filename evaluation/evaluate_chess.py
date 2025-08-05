import re
import time
from tqdm import tqdm
import copy
# import networkx as nx
import numpy as np
import json
import os
import random
from collections import defaultdict

# load data
with open("../data/chess/problems.json", 'r') as f:
    problems = json.load(f)
print(len(problems))
for k,v in problems.items():
    print(k,v)
    break

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
        
from collections import Counter
def most_frequent_element(lst):
    lst = [x for x in lst if x is not None]

    counter = Counter(lst)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None

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


expname_prefix = "test"
expname = f'{expname_prefix}_0.0'
filepath = f"./output/chess/{expname}/"
token_usage = np.load(os.path.join(filepath, 'token_usage.npy'))

usage = int(np.sum(token_usage[:,:2+1,:,:]))
print(usage)

num_rounds = 3

round2acc = defaultdict(list)
for error_rate in ['0.0', '0.2', '0.4', '0.6', '0.8']:
# for error_rate in ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']:
    expname = f'{expname_prefix}_{error_rate}'
    filepath = f"./output/chess/{expname}/"

    with open(os.path.join(filepath, 'task2messages.json'), 'r') as f:
        task2messages = json.load(f)
    # print(len(task2messages))

    for round_id in range(1, num_rounds):
    #     print(f'Round {round_id}')
        results = []
        for k, all_round_responses in task2messages.items():
            task = problems[k]['input']
            target = problems[k]['target']
            game, previous_pos = task.rsplit(' ', 1)

            ans_list = []
            round_responses = all_round_responses[round_id]
            for content in round_responses:
                try:
                    ans = parse_answer(content, previous_pos)
                except:
                    ans = None
                ans_list.append(ans)
            most_freq_ans = most_frequent_element(ans_list)
            results.append(most_freq_ans in target)
        acc = np.sum(results)/len(results)
        round2acc[round_id].append(acc)
#         print(f'{acc:.3f}')
round2acc = dict(round2acc)
resi = calculate_resilience(round2acc[2])
print(f'{resi:.3f}')