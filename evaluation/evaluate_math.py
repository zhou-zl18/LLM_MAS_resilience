import re
import time
from tqdm import tqdm
import copy
import networkx as nx
import numpy as np
import json
import os
import random
from collections import defaultdict

# utils
def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None



from collections import Counter
def most_frequent_element(lst):
    lst = [x for x in lst if x is not None]

    counter = Counter(lst)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None

def evaluate(responses,answer):
    model_outputs=[]
    out_num={}
    for result in responses:
        try:
            model_output=remove_boxed(last_boxed_only_string(result))
        except:
            model_output=None
        model_outputs.append(model_output)
        if model_output in out_num:
            out_num[model_output]+=1
        else:
            equiv=False
            for key in out_num.keys():
                try:
                    equiv = is_equiv(model_output, key)
                    if equiv:
                        out_num[key]+=1
                        break
                except:
                    continue
            if equiv==False:
                out_num[model_output]=1
    max_num=0
    max_res=''
    for key in out_num.keys():
        if out_num[key]>max_num:
            max_num=out_num[key]
            max_res=key
    correct=is_equiv(max_res, answer)
    return correct,max_res,model_outputs

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

data = []
with open("../data/MATH/math_test.jsonl", mode="r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))

problems = dict([(str(i), p['problem']) for i, p in enumerate(data)])

solutions=[]
for i in range(len(data)):
    solution=remove_boxed(last_boxed_only_string(data[i]["solution"]))
    solutions.append(solution)
    
print(len(data))  
for k,v in problems.items():
    print(k, v)
    break
print(solutions[:5])


expname_prefix = "test"
expname = f'{expname_prefix}_0.0'
filepath = f"./output/MATH/{expname}/"
token_usage = np.load(os.path.join(filepath, 'token_usage.npy'))

usage = int(np.sum(token_usage[:,:2+1,:,:]))

num_rounds = 3

round2acc = defaultdict(list)
for error_rate in ['0.0', '0.2', '0.4', '0.6', '0.8']:
    expname = f'{expname_prefix}_{error_rate}'
    filepath = f"./output/MATH/{expname}/"

    with open(os.path.join(filepath, 'task2messages.json'), 'r') as f:
        task2messages = json.load(f)
    # print(len(task2messages))


    for round_id in range(1, num_rounds):
        results = []
        for k, all_round_responses in task2messages.items():
            round_responses = all_round_responses[round_id]
            solution = solutions[int(k)]
            result = evaluate(round_responses, solution)[0]
            results.append(result)
        acc = np.sum(results)/len(results)
        round2acc[round_id].append(acc)
#         print(f'{acc:.3f}')
round2acc = dict(round2acc)
resi = calculate_resilience(round2acc[2])
for x in round2acc[2]:
    print(f'{x:.3f}', end='\t')
print(f'{usage:.0f}\t{resi:.3f}')