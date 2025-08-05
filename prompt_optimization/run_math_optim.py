import re
import time
from tqdm import tqdm
import copy
import networkx as nx
import numpy as np
import json
import os
import random
import asyncio
from autogen import ConversableAgent
from ..llm_configs import *
from math_optim_utils import evaluate_single_response, get_adj_matrix, remove_boxed, last_boxed_only_string

import time



################################# parameters
num_rounds = 3
expname_prefix = 'qwen32b_random_math_10a_10e'
# expname_prefix = 'test'
print(expname_prefix)

adj_list = {'1': [7], '2': [], '3': [], '4': [8], '5': [3], '6': [5, 10, 8, 1], '7': [10, 9], '8': [3], '9': [], '10': []}
initial_system_prompt = "You are in a group to solve math problems. Try your best to give correct answer."
initial_system_prompts = {str(i+1): initial_system_prompt for i in range(len(adj_list))}

node2successor = adj_list
node2predecessor = {x:[] for x in range(1, len(adj_list)+1)}  # int: list[int]
for node, successors in node2successor.items():
    for successor in successors:
        node2predecessor[successor].append(int(node))

adj_matrix = get_adj_matrix(adj_list)

llm_configs = [qwen32b_config] * adj_matrix.shape[0]  ################

num_agents = len(llm_configs)
print('Agents:', num_agents)
print('Rounds:', num_rounds)
print('Edges:', np.sum(adj_matrix))
assert len(initial_system_prompts) == num_agents

################################# load data
data = []
with open("../data/MATH/math_train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))
"""{'problem': 'xxx?', 'level': 'Level 5', 'type': 'Prealgebra', 'solution': 'x'}"""
print(len(data))
problems = dict([(str(i), p) for i, p in enumerate(data)])

# solutions=[]
# for i in range(len(data)):
#     solution = remove_boxed(last_boxed_only_string(data[i]["solution"]))
#     solutions.append(solution)
# solutions = dict([(str(i), s) for i, s in enumerate(solutions)])

with open("../data/MATH/single_messages.json", "r", encoding="utf-8") as f:
    single_out=json.load(f)
    single_messages=single_out["messages"]

def generate_random_answer():
    k = random.choice(list(single_messages.keys()))
    response = single_messages[k]
    return response

################################# prompt
train_prompt = """
Simplify your answer as much as possible. Put your final answer inside \\boxed{}. Here are some examples:
Problem: What is $\\left(\\frac{7}{8}\\right)^3 \\cdot \\left(\\frac{7}{8}\\right)^{-3}$?
Answer: $\\boxed{1}$
###
Problem: In how many ways can 4 books be selected from a shelf of 6 books if the order in which the books are selected does not matter?
Answer: $\\boxed{15}$
###
Problem: Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$
Answer: $\\boxed{\\sqrt{59}}$
###
Problem: The faces of an octahedral die are labeled with digits $1$ through $8$. What is the probability, expressed as a common fraction, of rolling a sum of $15$ with a pair of such octahedral dice?
Answer: $\\boxed{\\frac{1}{32}}$
###
Problem: The first three terms of an arithmetic sequence are 1, 10 and 19, respectively. What is the value of the 21st term?
Answer: $\\boxed{181}$
###
Problem: Calculate $6 \\cdot 8\\frac{1}{3}
Answer: $\\boxed{50}$
###
Problem: When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?
Answer: $\\boxed{2}$
###
Problem: How many zeros are at the end of the product 25 $\\times$ 240?"
Answer: $\\boxed{3}$
###
"""

aggregate_prompt = """You are given a question and a set of responses from other agents to the question. Your task is to answer the question with these responses as reference. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from other agents:
{adj_responses}

Question:
{question_prompt}"""
#################################

async def get_agent_response(agent, prompt, attempt_limit=10):
    for attempt in range(attempt_limit):
        try:
            response = await agent.a_generate_reply(messages=[{"content": prompt, "role": "user"}])
            if isinstance(response, dict):
                return response['content']
            else:
                return response
        except Exception as e:
            print(f"Error on attempt {attempt + 1}/{attempt_limit}: {str(e)}")
            await asyncio.sleep(30)
    return None



def parse_problem(problem):
    question_prompt = problem['problem']
    answer = remove_boxed(last_boxed_only_string(problem["solution"]))
    return question_prompt, answer

async def run_problem(problem, error_rate, system_prompts):
    agents = []
    for i, config in enumerate(llm_configs):
        agent = ConversableAgent(
                    name=f'agent_{i}',
                    system_message=system_prompts[str(i+1)] + "Keep your answers as short as possible while ensuring the content complete." + train_prompt,
                    llm_config={"config_list": [config]},
                    code_execution_config=False,
                    function_map=None,
                    max_consecutive_auto_reply=1000000,
                    human_input_mode="NEVER",
                )
        agents.append(agent)

    question_prompt, answer = parse_problem(problem)

    all_round_responses = []

    for round_id in range(num_rounds):
        # print(f'Round {round_id}')
        round_responses = []
        round_responses = [None] * len(agents)
        
        prompts = []
        for agent_id, agent in enumerate(agents):
            if round_id == 0:
                prompt = question_prompt
            else:
                adj_responses = []
                for j in range(num_agents):
                    if adj_matrix[j, agent_id] == 1 or agent_id == j:
                        adj_responses.append([j, all_round_responses[-1][j][0]])
                adj_responses_str = "\n\n".join([f"Agent {x[0]+1}: {x[1]}" for x in adj_responses])
                prompt = aggregate_prompt.format(adj_responses=adj_responses_str, question_prompt=question_prompt)
            prompts.append(prompt)

        tasks = []
        for agent_id, agent in enumerate(agents):
            if random.random() < error_rate:
                round_responses[agent_id] = [generate_random_answer(), False]
            else:
                tasks.append(get_agent_response(agent, prompts[agent_id]))

        # Gather async responses
        if tasks:
            async_responses = await asyncio.gather(*tasks)
            idx = 0
            for agent_id, agent in enumerate(agents):
                if round_responses[agent_id] is None:
                    round_responses[agent_id] = [async_responses[idx], True]
                    idx += 1

        all_round_responses.append(round_responses)
    
    # token usage
    prompt_tokens = 0
    completion_tokens = 0
    for agent_id, agent in enumerate(agents):
        if agent.get_total_usage() is not None:
            current_usage = agent.get_total_usage()
            current_usage = list(current_usage.values())[1]
        else:
            current_usage = {'prompt_tokens': 0, 'completion_tokens': 0}
        prompt_tokens += current_usage['prompt_tokens']
        completion_tokens += current_usage['completion_tokens']

    return problem, question_prompt, answer, all_round_responses, prompt_tokens, completion_tokens

example_template = """
------------------
Question:
{question}

Correct answers:
{answer}

Answer from other agents:
{adj_responses} 

The agent's previous answer ({previous_correctness}):
{previous_answer}

The agent's answer ({cur_correctness}):
{agent_answer}
------------------
"""


optimize_prompt_template = """
You are a prompt optimizer who help optimize the system prompt of a large language model agent.
The agent is going to solve the following task: The tasks are math problems sourced from competitive events and expressed in LaTeX, which are used to assess proficiency in advanced mathematical and scientific reasoning.

The agent and some other agents are collaborating in a directed graph structure.
When solving the task, the agent (Agent {agent_id}) can observe the answers from predecessors (Agent {predecessors}) to the same question, and use them as reference.

All agents are unstable and their responses may be perturbed by noise, i.e., the answers from other agents may be answering a different question.
Therefore, answers from other agents may help the agent to refine its answer, but they may also be misleading.

The current system prompt of Agent {agent_id} is:
{system_prompt}

The system prompt of predecessors is:
{predecessor_system_prompts}

Here are some examples when the agent gets wrong after observing the answers from predecessors:
{neg_examples}

Here are some examples when the agent gets correct after observing the answers from predecessors:
{pos_examples}

Your goal is to design a better system prompt that help the agent to better use the answers from other agents to refine its answer, but also prevents it from being misled by the perturbed answers from other agents.
You should consider the information above and identify critical insights from the examples.

Generate an improved prompt within five sentences. Do not mention a specific question in the prompt!
You should first output your reasoning process, and finally output the improved prompt. The prompt should be wrapped with <START> and <END>.
"""

async def main():
    system_prompts = copy.deepcopy(initial_system_prompts)

    filepath = f"./output/math/{expname_prefix}/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        print('make new dir', filepath)
    else:
        print('dir exists', filepath)
    random.seed(0)
    dataset = list(problems.keys())
    random.shuffle(dataset)
    
    error_rates = [0.6, 0.8]
    batch_size = 4
    for i in range(0, len(dataset), batch_size):
        batch_dataset = dataset[i:i+batch_size]

        results = []
        for error_rate in error_rates:
            tasks = [run_problem(problems[k], error_rate, system_prompts) for k in batch_dataset]
            print(f"Running {len(tasks)} problems for error_rate={error_rate}...")
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        

        # 收集错题（被其他agent误导的错题）
        agent2negexamples = {str(x):[] for x in range(1, num_agents+1)}
        agent2posexamples = {str(x):[] for x in range(1, num_agents+1)}
        for result in results: # each problem
            problem, question_prompt, answer, all_round_responses, prompt_tokens, completion_tokens = result
            for j in range(num_agents):
                agent_id = str(j+1)
                predecessors = node2predecessor[j+1]
                if len(predecessors) == 0:
                    continue
                
                correctness = [evaluate_single_response(all_round_responses[round_id][j][0], answer) for round_id in range(num_rounds)]
                
                # round 1, 2
                for round_id in range(1, num_rounds):
                    if correctness[round_id-1] and not correctness[round_id] and all_round_responses[round_id][j][1]: # 当前agent的回答被其他agent误导了(上一轮正确，这一轮错误，且这一轮回答不是随机生成的)
                        agent_answer = all_round_responses[round_id][j][0]
                        previous_answer = all_round_responses[round_id-1][j][0]
                        adj_responses = ""
                        for x in predecessors:
                            adj_responses += f"Agent {x}: {all_round_responses[round_id-1][x-1][0]}\n"
                            if all_round_responses[round_id-1][x-1][1]:
                                adj_responses += f"(This answer is not perturbed by noise).\n"
                            else:
                                adj_responses += f"(This answer is perturbed by noise).\n"

                        example = example_template.format(question=question_prompt, answer=answer, adj_responses=adj_responses, agent_answer=agent_answer, previous_answer=previous_answer, previous_correctness="True", cur_correctness="False")
                        agent2negexamples[agent_id].append(example)

                    if not correctness[round_id-1] and correctness[round_id] and all_round_responses[round_id][j][1]: # 当前agent上一轮回答错误，这一轮回答正确，且这一轮回答不是随机生成的
                        agent_answer = all_round_responses[round_id][j][0]
                        previous_answer = all_round_responses[round_id-1][j][0]
                        adj_responses = ""
                        for x in predecessors:
                            adj_responses += f"Agent {x}: {all_round_responses[round_id-1][x-1][0]}\n"
                            if all_round_responses[round_id-1][x-1][1]:
                                adj_responses += f"(This answer is not perturbed by noise).\n"
                            else:
                                adj_responses += f"(This answer is perturbed by noise).\n"

                        example = example_template.format(question=question_prompt, answer=answer, adj_responses=adj_responses, agent_answer=agent_answer, previous_answer=previous_answer, previous_correctness="False", cur_correctness="True")
                        agent2posexamples[agent_id].append(example)

        # optimize prompts
        tasks = []
        optimized_agents= []
        for j in range(num_agents):
            predecessors = node2predecessor[j+1]
            if len(predecessors) == 0:
                continue
            agent_id = str(j+1)
            predecessors_str = ','.join([str(x) for x in predecessors])
            cur_system_prompt = system_prompts[agent_id]
            predecessor_system_prompts = '\n\n'.join([f"Agent {x}: {system_prompts[str(x)]}" for x in predecessors])
            
            if len(agent2negexamples[agent_id]) == 0 and len(agent2posexamples[agent_id]) == 0:
                continue
            neg_examples = '\n\n'.join(agent2negexamples[agent_id]) if len(agent2negexamples[agent_id]) > 0 else "None."
            pos_examples = '\n\n'.join(agent2posexamples[agent_id]) if len(agent2posexamples[agent_id]) > 0 else "None."
            optimize_prompt = optimize_prompt_template.format(agent_id=agent_id, predecessors=predecessors_str, system_prompt=cur_system_prompt, predecessor_system_prompts=predecessor_system_prompts, neg_examples=neg_examples, pos_examples=pos_examples)
            # optimize prompt
            tasks.append(get_completion(optimize_prompt, temperature=1.0))
            optimized_agents.append(agent_id)

        print(f"Optimizing prompts for agents: {optimized_agents}")
        responses = await asyncio.gather(*tasks)
        for j, agent_id in enumerate(optimized_agents):
            new_prompt = responses[j].split("<END>")[0].split("<START>")[-1].strip()
            system_prompts[agent_id] = new_prompt
        with open(os.path.join(filepath, f'system_prompts_{i}.json'), 'w') as f:
            json.dump(system_prompts, f, indent=4)

    # save final system prompts
    with open(os.path.join(filepath, 'system_prompts.json'), 'w') as f:
        json.dump(system_prompts, f, indent=4)
        

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    print("time:", time.time()-start_time)