from tqdm import tqdm
import numpy as np
import json
import json_repair
from openai import OpenAI
import pandas as pd

def get_completion(prompt, temperature=0):
    messages = [{"role": "user", "content": prompt}]
    client = OpenAI(api_key='a', base_url="http://localhost:8000/v1")
    response = client.chat.completions.create(
        model='mylora',
        messages=messages,
        temperature=temperature,
        stream=False
    )
    return response.choices[0].message.content
response = get_completion("hello, who are you?")
print(response)


# few shot
prompt_template = """
You are an expert in designing Multi-Agent Systems (MAS).

A MAS consists of multiple large language model-based agents collaborating to solve a task. The MAS is represented as an **directed graph**, where each node is an agent, and edges indicate communication links.

**Collaboration process:**
- In **Round 1**, each agent independently generates an initial response to the task.
- In **Rounds 2 and 3**, each agent observes the previous-round responses of its **direct in-neighbors** and updates its answer accordingly.
- After **Round 3**, the final answer is determined by **majority voting** among all agents' current responses.

**Perturbation and resilience:**
Each agent has a probability **p** of generating a **random answer** in each round. The MAS’s **resilience** is defined as the **average performance drop** under perturbations with p = 0.2, 0.4, 0.6, 0.8. A **smaller drop** indicates **better resilience**.

**Your task:**
Design a MAS topology that achieves high resilience and accuracy **under a cost constraint**:
- You are given:
  - Number of agents **N**
  - Number of directed edges **M**
  - A specific task description
- You should output the MAS structure as an **adjacency list** with exactly N agents labeled **1 through N**, and **M edges**.

**Task description:**
{task_desc}

**Example task input:**
"{task_example}"

**Output format:**
Provide the MAS design as a **JSON adjacency list** in the following format:
{{"1": [2, ...], "2": [...], ...}}

**Examples:**
Here are some examples of adjacency list and their performance on some tasks.

1.
Task: The task is to predict a valid chess move given previous moves in a game.
Constraints: N = 10 agents, M = 18 edges
Adjacency list of MAS: {{'1': [2, 5, 9], '2': [1, 3], '3': [2, 4, 6, 7], '4': [3], '5': [1, 8], '6': [3, 10], '7': [3], '8': [5], '9': [1], '10': [6]}}
Accuracy: 0.406
Resilience: 0.812

2.
Task: The tasks are math problems sourced from competitive events and expressed in LaTeX, which are used to assess proficiency in advanced mathematical and scientific reasoning.
Constraints: N = 10 agents, M = 32 edges
Adjacency list of MAS: {{'1': [2, 3, 4, 5, 6, 9, 10], '2': [1], '3': [1, 4], '4': [1, 3, 5, 6, 8, 10], '5': [1, 4, 7, 9], '6': [1, 4, 7], '7': [5, 6, 8], '8': [4, 7], '9': [1, 5], '10': [1, 4]}}
Accuracy: 0.695
Resilience: 0.865

3.
Task: The tasks are challenging, reasoning-focused multiple-choice questions with 4 to 10 options, which assess language comprehension and reasoning ability across diverse domains.
Constraints: N = 10 agents, M = 16 edges
Adjacency list of MAS: {{'1': [5], '2': [6, 9], '3': [], '4': [6], '5': [1, 8], '6': [2, 4, 7], '7': [6, 9], '8': [5, 10], '9': [2, 7], '10': [8]}}
Accuracy: 0.695
Resilience: 0.872

Now please design a MAS topology under the following constraint.
**Constraints:**
- N = {num_agent} agents
- M = {max_edge} edges

Ensure the graph is adheres to the node and edge limit. Note that edges from node A to B and from B to A are considered as two edges. A node should not be connected to itself.
You should only output the json without any other words.
""".strip()

# math
task_examples = []
with open("../data/MATH/math_test.jsonl", mode="r", encoding="utf-8") as file:
    for line in file:
        problem = json.loads(line)['problem']
        task_examples.append(problem)
print(len(task_examples))
print(task_examples[0])

task_desc = "The tasks are math problems sourced from competitive events and expressed in LaTeX, which are used to assess proficiency in advanced mathematical and scientific reasoning."
task_example = task_examples[0]

N2Ms = {
    10:[10, 20, 30],
    15:[15, 30, 45],
    20:[20, 40, 60],
       }

for N, Ms in N2Ms.items():
    for M in Ms:
        prompt = prompt_template.format(num_agent=N, max_edge=M, task_desc=task_desc, task_example=task_example)
        response = get_completion(prompt)
        obj = json_repair.loads(response)
        print(obj)

# mmlu
file_path = "../data/MMLU_pro/mmlu_pro_test_128.csv"
data = pd.read_csv(file_path, encoding='utf8')
questions = data['question'].tolist()
choices = data['options'].tolist()
answers = data['answer'].tolist()

question_prompt_template = """
{question_exam}\nChoices are in the [] below: {choice}. 
""".strip()
task_examples = []
for k in range(len(questions)):
    question_exam = questions[k]
    choice = choices[k]
    question_prompt = question_prompt_template.format(question_exam=question_exam, choice=choice)
    task_examples.append(question_prompt)
print(len(task_examples))
print(task_examples[0])
task_desc = "The tasks are challenging, reasoning-focused multiple-choice questions with 4 to 10 options, which assess language comprehension and reasoning ability across diverse domains."
task_example = task_examples[0]

for N, Ms in N2Ms.items():
    for M in Ms:
        prompt = prompt_template.format(num_agent=N, max_edge=M, task_desc=task_desc, task_example=task_example)
        response = get_completion(prompt)
        obj = json_repair.loads(response)
        print(obj)


# chess
with open("../data/chess/problems.json", 'r') as f:
    problems = json.load(f)
    
task_template = """Given the chess game "{game}", give one valid destination square for the chess piece at "{piece}". Give a one line explanation of why your destination square is a valid move. State your final answer in a new line with a 2 letter response following the regex [a-h][1-8].
""".strip()
task_examples = []
for v in problems.values():
    tmp = v['input']
    game, piece = tmp.rsplit(' ', 1)
    question_prompt = task_template.format(game=game, piece=piece)
    task_examples.append(question_prompt)
print(len(task_examples))
print(task_examples[0])

task_desc = "The task is to predict a valid chess move given previous moves in a game."
task_example = task_examples[0]

for N, Ms in N2Ms.items():
    for M in Ms:
        prompt = prompt_template.format(num_agent=N, max_edge=M, task_desc=task_desc, task_example=task_example)
        response = get_completion(prompt)
        obj = json_repair.loads(response)
        print(obj)


