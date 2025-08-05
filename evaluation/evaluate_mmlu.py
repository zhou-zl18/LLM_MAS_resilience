import re
import time
from tqdm import tqdm
import copy
# import networkx as nx
import numpy as np
import json
import os
import random
import pandas as pd
from collections import defaultdict

file_path = '../data/MMLU_pro/mmlu_pro_test_128.csv'
data = pd.read_csv(file_path, encoding='utf8')
print(data.shape)

questions = data['question'].tolist()
choices = data['options'].tolist()
answers = data['answer'].tolist()

def parse_answer(content:str):
    content = content.strip()
    return content[-1].upper()

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
filepath = f"./output/MMLU/{expname}/"
token_usage = np.load(os.path.join(filepath, 'token_usage.npy'))

usage = int(np.sum(token_usage[:,:2+1,:,:]))

num_rounds = 3

round2acc = defaultdict(list)
for error_rate in ['0.0', '0.2', '0.4', '0.6', '0.8']:
    expname = f'{expname_prefix}_{error_rate}'
    filepath = f"./output/MMLU/{expname}/"

    with open(os.path.join(filepath, 'task2messages.json'), 'r') as f:
        task2messages = json.load(f)
  
    for round_id in range(0, num_rounds):
        results = []
        for k, all_round_responses in task2messages.items():
            k = int(k)
            task = questions[k]
            target = answers[k]
            ans_list = []
            round_responses = all_round_responses[round_id]
            for content in round_responses:
                ans = parse_answer(content)
                ans_list.append(ans)
            most_freq_ans = most_frequent_element(ans_list)
            results.append(most_freq_ans in target)
        acc = np.sum(results)/len(results)
        round2acc[round_id].append(acc)
round2acc = dict(round2acc)
resi = calculate_resilience(round2acc[2])
for x in round2acc[2]:
    print(f'{x:.3f}', end='\t')
print(f'{usage:.0f}\t{resi:.3f}')