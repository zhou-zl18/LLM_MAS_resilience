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
from chess_optim_utils import calculate_acc, calculate_resilience, evaluate_single_response, get_adj_matrix

import time
start_time = time.time()


################################# parameters
num_rounds = 3
expname_prefix = 'qwen32b_random_chess_10a_10e'
# expname_prefix = 'test'
print(expname_prefix)

adj_list = {'1': [7], '2': [], '3': [], '4': [8], '5': [3], '6': [5, 10, 8, 1], '7': [10, 9], '8': [3], '9': [], '10': []}
initial_system_prompt = "You are an expert skilled in playing chess."
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

# multi agent时temperature设为1
temperature = 1.0
for c in llm_configs:
    c['temperature'] = temperature

################################# load data
with open("../data/chess/train_problems.json", 'r') as f:
    problems = json.load(f)

# problems = dict(list(problems.items())[:5])
# print(len(problems))

with open('../data/chess/task2response.json', 'r') as f:
    single_task2messages = json.load(f)

def generate_random_answer():
    k = random.choice(list(single_task2messages.keys()))
    response = single_task2messages[k]
    return response

################################# prompt
prompt_template = """Given the chess game "{game}", give one valid destination square for the chess piece at "{piece}". Give a one line explanation of why your destination square is a valid move. State your final answer in a new line with a 2 letter response following the regex [a-h][1-8].
""".strip()
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
    tmp = problem['input']
    game, piece = tmp.rsplit(' ', 1)
    question_prompt = prompt_template.format(game=game, piece=piece)
    answer = problem['target']
    return question_prompt, answer

async def run_problem(problem, error_rate, system_prompts):
    agents = []
    for i, config in enumerate(llm_configs):
        agent = ConversableAgent(
                    name=f'agent_{i}',
                    system_message=system_prompts[str(i+1)],
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

async def run_chess(error_rate, system_prompts):
    expname = f'{expname_prefix}_{error_rate}'
    print('Error rate:', error_rate)

    # filepath = f"../output/chess_train/{expname}/"
    # if not os.path.exists(filepath):
    #     os.makedirs(filepath)
    #     print('make new dir',filepath)

    random.seed(0)
    task2messages = {}

    total_prompt_tokens = 0
    total_completion_tokens = 0
    for k,v in tqdm(list(problems.items()), desc=f'Error_rate_{error_rate}'):
        problem, question_prompt, answer, all_round_responses, prompt_tokens, completion_tokens = await run_problem(v, error_rate, system_prompts) 
        task2messages[k] = all_round_responses
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

    # with open(os.path.join(filepath, 'task2messages.json'), 'w') as f:
    #     json.dump(task2messages, f, indent=4)
    # print("time:", time.time()-start_time)
    # print('Saved to', filepath)

    acc = calculate_acc(problems, task2messages, round_id=2)
    usage = total_prompt_tokens + total_completion_tokens
    return [acc, usage]


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
The agent is going to solve the following task: predict a valid chess move given previous moves in a game. Note that it is not playing against an opponent. The only goal is to give a **valid** move instead of to win.

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

    filepath = f"./output/chess/{expname_prefix}/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        print('make new dir', filepath)
    else:
        print('dir exists', filepath)
    random.seed(0)
    dataset = list(problems.keys())
    random.shuffle(dataset)
    
    error_rates = [0.6, 0.8]
    batch_size = 8
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
                
                correctness = [evaluate_single_response(problem, all_round_responses[round_id][j][0]) for round_id in range(num_rounds)]
                
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

                    if not correctness[round_id-1] and all_round_responses[round_id-1][j][1] and correctness[round_id] and all_round_responses[round_id][j][1]: # 当前agent上一轮回答错误，这一轮回答正确，且这一轮回答不是随机生成的
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
        

    # tasks = [run_chess(error_rate, dataset) for error_rate in [0.0, 0.2, 0.4, 0.6, 0.8]]
    # results = await asyncio.gather(*tasks)
    # accs = [result[0] for result in results]
    # resilience = calculate_resilience(accs)
    # acc = accs[0]
    # usage = results[0][1]
    
    # for x in accs:
    #     print(f'{x:.3f}', end='\t')
    # print(f'{usage:.0f}\t{resilience:.3f}')

    # return acc, resilience, usage

if __name__ == "__main__":
    asyncio.run(main())