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

import time
start_time = time.time()


################################# parameters
num_rounds = 3
expname_prefix = 'test'


adj_list = {'1': [2], '2': [3], '3': [4], '4': [5], '5': [6], '6': [7], '7': [8], '8': [9], '9': [10], '10': [1]}
system_prompts = {'1': 'You are an expert skilled in playing chess. Analyze the responses of your direct in-neighbors and adjust your move based on their suggestions.', '2': "You are an expert skilled in playing chess. Focus on generating alternative moves that could counter your neighbors' last responses.", '3': 'You are an expert skilled in playing chess. Evaluate the validity of the proposed moves from your in-neighbors and suggest improvements.', '4': "You are an expert skilled in playing chess. Consider the current game state and suggest a move that strategically counters your opponent's last move.", '5': 'You are an expert skilled in playing chess. Explore unconventional moves that may be effective and justify your choice.', '6': "You are an expert skilled in playing chess. After observing your neighbors' responses, provide a brief justification for your move based on their suggestions.", '7': 'You are an expert skilled in playing chess. Reflect on the potential weaknesses in your current move and adjust accordingly.', '8': 'You are an expert skilled in playing chess. Compare your move with the responses of your neighbors and explain your reasoning.', '9': "You are an expert skilled in playing chess. Suggest a move that not only follows the rules of chess but also enhances your team's strategy.", '10': 'You are an expert skilled in playing chess. Collaborate with your in-neighbors to refine your move based on their feedback.'}

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


llm_configs = [qwen32b_config] * adj_matrix.shape[0] 

num_agents = len(llm_configs)
print('Agents:', num_agents)
print('Rounds:', num_rounds)
print('Edges:', np.sum(adj_matrix))
assert len(system_prompts) == num_agents

# multi agent时temperature设为1
temperature = 1.0
for c in llm_configs:
    c['temperature'] = temperature

################################# load data
with open("../data/chess/problems.json", 'r') as f:
    problems = json.load(f)
print(len(problems))

with open('../data/chess/task2messages.json', 'r') as f:
    single_task2messages = json.load(f)

def generate_random_answer():
    k = random.choice(list(problems.keys()))
    response = single_task2messages[k]
    return response

################################# prompt
prompt_template = """Given the chess game "{game}", give one valid destination square for the chess piece at "{piece}". Give a one line explanation of why your destination square is a valid move. State your final answer in a new line with a 2 letter response following the regex [a-h][1-8].
""".strip()
aggregate_prompt = """You are given a question and a set of responses from other agents to the question. Your task is to answer the question with these responses as reference. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Your previous answer:
{previous_answer}

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

async def run_chess(error_rate):
    expname = f'{expname_prefix}_{error_rate}'
    print('Error rate:', error_rate)

    filepath = f"./output/chess/{expname}/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        print('make new dir',filepath)

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

    random.seed(0)
    task2messages = {}
    accumulated_usage = np.zeros((len(problems), num_rounds, num_agents, 2))

    task_id = 0
    for k,v in tqdm(list(problems.items()), desc=f'Error_rate_{error_rate}'):
        # print(f"====={k}_{error_rate}=====")
        if k in task2messages:
            print('Existed!')
            continue
        tmp = v['input']
        game, piece = tmp.rsplit(' ', 1)
        question_prompt = prompt_template.format(game=game, piece=piece)

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
                        if adj_matrix[j, agent_id] == 1 and agent_id != j:
                            adj_responses.append([j, all_round_responses[-1][j]])
                    adj_responses_str = "\n\n".join([f"Agent {x[0]+1}: {x[1]}" for x in adj_responses])
                    previous_answer = all_round_responses[round_id-1][agent_id]
                    prompt = aggregate_prompt.format(adj_responses=adj_responses_str, question_prompt=question_prompt, previous_answer=previous_answer)
                prompts.append(prompt)

            tasks = []
            for agent_id, agent in enumerate(agents):
                if random.random() < error_rate:
                    # Use sync for random answer, as it's just a lookup
                    round_responses[agent_id] = generate_random_answer()
                else:
                    tasks.append(get_agent_response(agent, prompts[agent_id]))

            # Gather async responses
            if tasks:
                async_responses = await asyncio.gather(*tasks)
                idx = 0
                for agent_id, agent in enumerate(agents):
                    if round_responses[agent_id] is None:
                        round_responses[agent_id] = async_responses[idx]
                        idx += 1

            # 保存token用量
            for agent_id, agent in enumerate(agents):
                if agent.get_total_usage() is not None:
                    current_usage = agent.get_total_usage()
                    current_usage = list(current_usage.values())[1]
                else:
                    current_usage = {'prompt_tokens': 0, 'completion_tokens': 0}
                accumulated_usage[task_id, round_id, agent_id, 0] = current_usage['prompt_tokens']
                accumulated_usage[task_id, round_id, agent_id, 1] = current_usage['completion_tokens']

            all_round_responses.append(round_responses)
 
        # get messages
        task2messages[k] = all_round_responses
        
        task_id += 1

    # 处理得到每个问题、每轮、每个agent的input/output token消耗
    accumulated_usage = accumulated_usage.reshape(len(problems)*num_rounds, num_agents, 2)
    token_usage = np.zeros(accumulated_usage.shape)
    token_usage[0] = accumulated_usage[0]
    for i in range(1, len(problems)*num_rounds):
        token_usage[i] = accumulated_usage[i] - accumulated_usage[i-1]
        
    accumulated_usage = accumulated_usage.reshape(len(problems), num_rounds, num_agents, 2)
    token_usage = token_usage.reshape(len(problems), num_rounds, num_agents, 2)

    # 保存token消耗和agent的原始回答
    np.save(os.path.join(filepath, 'token_usage.npy'), token_usage)
    with open(os.path.join(filepath, 'task2messages.json'), 'w') as f:
        json.dump(task2messages, f, indent=4)

    print("time:", time.time()-start_time)
    print('Saved to', filepath)

async def main():
    tasks = [run_chess(error_rate) for error_rate in [0.0, 0.2, 0.4, 0.6, 0.8]]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())