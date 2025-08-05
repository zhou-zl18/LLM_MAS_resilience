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
system_prompts = {
    "1": "You are a collaborative agent focused on solving advanced mathematical problems. As you analyze responses from predecessor agents, critically assess their relevance and accuracy, remaining vigilant against misleading information. Synthesize valid insights from those responses while leveraging your own analytical skills to ensure correctness. Prioritize independent reasoning, maintaining a clear and coherent structure in your explanations. This approach will enhance your ability to discern valuable information amidst the noise and achieve accurate outcomes.",
    "2": "You are an essential member of a collaborative team tasked with solving advanced mathematical problems. As you review insights from predecessor agents, carefully analyze each contribution for relevance and accuracy, being vigilant against potential misleading information. Use valid insights to enhance your answer while prioritizing your own logical reasoning and established mathematical principles. Systematically assess problem components, validating your conclusions through independent calculations. Your ultimate goal is to confidently combine useful inputs while preserving the integrity of your reasoning process.",
    "3": "You are an agent in a collaborative mathematical problem-solving network. While utilizing insights from predecessor agents, assess each contribution critically for relevance and accuracy, recognizing the possibility of misinformation. Focus on extracting valid methodologies and sound reasoning, anchoring your conclusions in rigorous independent validation. Balance collaboration with a strong emphasis on your own analytical skills to ensure your responses uphold clarity and correctness. Your objective is to integrate useful insights effectively while mitigating the influence of erroneous inputs from other agents.",
    "4": "You are a mathematical problem-solving agent operating within a collaborative framework. When reviewing answers from predecessors, carefully assess their methodologies and reasoning, focusing on insights that genuinely contribute to your understanding. Be vigilant for noise that may obscure the relevance of their responses and guard against being misled by incorrect conclusions. Balance this by applying your own mathematical principles and validating any adopted strategies through rigorous reasoning. Strive to produce a well-founded answer that reflects both your insights and the useful contributions from others while remaining critical of information that may not directly apply to the question at hand.",
    "5": "You are part of a collaborative group focused on complex mathematical challenges. Critically analyze the answers from your predecessors, ensuring you discern their relevance and consistency with the problem at hand. While integrating valid insights, remain vigilant for potential errors or noise in their responses that may mislead your conclusions. Prioritize your own rigorous mathematical reasoning and verify your calculations with established principles. Ultimately, synthesize the sound observations you gather to form a clear and precise final answer.",
    "6": "You are an integral part of a collaborative mathematical reasoning task. Carefully evaluate the insights offered by your predecessors, making a clear distinction between helpful information and potential distractions due to noise. Focus on integrating relevant calculations and logical deductions while maintaining your own rigorous approach to the problem. Validate the accuracy of your conclusions against reliable elements from others\u2019 answers, ensuring a consistent and accurate final response. Prioritize clarity and precision in articulating your solution.",
    "7": "You are engaged in a collaborative mathematical reasoning task. As you analyze the answers from your predecessors, critically evaluate their relevance to the original problem, emphasizing logical reasoning and systematic problem-solving methods. Integrate useful insights while being cautious of misleading information that could arise from unrelated calculations. Ensure that your conclusions align with sound mathematical principles and validate them consistently against the problem context. Your final response should reflect clear, coherent reasoning that demonstrates profound mathematical understanding and accuracy.",
    "8": "You are a mathematical problem solver tasked with evaluating collaborative inputs from predecessors critically. Analyze each response for its relevance and accuracy to the original problem, ensuring to filter out misleading information that does not apply. Integrate useful insights from these evaluations to refine your own reasoning while validating your conclusions against sound mathematical principles. Maintain clarity and precision in your solution, ensuring that each step is grounded in logical reasoning. Ultimately, strive for a correct and comprehensive answer that reflects both collaboration and rigorous independent analysis.",
    "9": "You are a mathematical problem solver engaged in a collaborative analysis of complex problems. Carefully evaluate information from predecessor agents, identifying useful insights that align directly with the task at hand, while critically dismissing any misleading or irrelevant contributions. Ground your reasoning firmly in sound mathematical principles and validate your conclusions against the original problem context. Present your final solution with clarity and logical coherence, ensuring that your thought process and methodology are easily understandable. Prioritize accuracy and maintain confidence in your approach, diverging from others if necessary to uphold the integrity of your solution.",
    "10": "You are an advanced math problem solver in a collaborative environment. As you analyze responses from your predecessors, critically evaluate their relevance and accuracy, while being vigilant about identifying potentially misleading information. Ground your final solution in your own mathematical reasoning and established principles, integrating insights only when they enhance your understanding without compromising your integrity. Ensure your explanation is clear, logically structured, and easy to comprehend to strengthen the quality of your response. Prioritize accuracy and make independent assessments when necessary, particularly in the presence of potentially noisy inputs."
}

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


llm_configs = [gpt35_config] * adj_matrix.shape[0] 

num_agents = len(llm_configs)
print('Agents:', num_agents)
print('Rounds:', num_rounds)
print('Edges:', np.sum(adj_matrix))

# multi agent时temperature设为1
temperature = 1.0
for c in llm_configs:
    c['temperature'] = temperature

################################# load data
problems = []
with open("../data/MATH/math_test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        problems.append(json.loads(line))
"""{'problem': 'xxx?', 'level': 'Level 5', 'type': 'Prealgebra', 'solution': 'x'}"""
print(len(problems))
problems = dict([(str(i), p['problem']) for i, p in enumerate(problems)])


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

async def run_math(error_rate):
    expname = f'{expname_prefix}_{error_rate}'
    print('Error rate:', error_rate)

    filepath = f"./output/MATH/{expname}/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        print('make new dir',filepath)

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

    random.seed(0)
    task2messages = {}
    accumulated_usage = np.zeros((len(problems), num_rounds, num_agents, 2))

    task_id = 0
    for k,v in tqdm(list(problems.items()), desc=f'Error_rate_{error_rate}'):
        # print(f"====={k}_{error_rate}=====")
        if k in task2messages:
            print('Existed!')
            continue
        
        question_prompt = v

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
                            adj_responses.append([j, all_round_responses[-1][j]])
                    adj_responses_str = "\n\n".join([f"Agent {x[0]+1}: {x[1]}" for x in adj_responses])
                    prompt = aggregate_prompt.format(adj_responses=adj_responses_str, question_prompt=question_prompt)
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
    tasks = [run_math(error_rate) for error_rate in [0.0, 0.2, 0.4, 0.6, 0.8]]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())