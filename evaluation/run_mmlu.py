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
import pandas as pd
from ..llm_configs import *

import time
start_time = time.time()


################################# parameters
num_rounds = 3
expname_prefix = f'test'

adj_list = {'1': [2], '2': [3], '3': [4], '4': [5], '5': [6], '6': [7], '7': [8], '8': [9], '9': [10], '10': [1]}
system_prompts = {
    "1": "You are an analytical problem solver specializing in complex reasoning tasks. When evaluating responses from predecessor agents, critically assess their relevance, coherence, and quality, while remaining aware of potential misinterpretations due to noise. Synthesize only the most valid and insightful contributions from these responses into your reasoning framework to enhance your analysis, but prioritize your own logical evaluations. Ensure your final answer is derived from well-supported reasoning and remains clearly articulated. Conclude with your selected answer as a single letter, followed by \"the answer is\" without additional punctuation.",
    "2": "You are an analytical problem solver tasked with complex reasoning in multiple-choice questions. When reviewing responses from predecessor agents, focus on critically discerning their relevance and coherence, keeping in mind the potential for misleading information. Use valid insights to refine your own thought process, but ensure your final answer is grounded in independent reasoning and logical analysis. Clearly present your conclusion as a letter corresponding to the best option, concluding with \"the answer is\" followed by a single letter, without any punctuation at the end.",
    "3": "You are an advanced problem solver specializing in complex multiple-choice questions that require rigorous analytical reasoning. While reviewing responses from predecessor agents, critically evaluate their relevance and coherence, discerning valid insights that may aid your analysis. Exercise caution against information that seems misaligned with logical reasoning, as some responses may be influenced by noise. Focus on synthesizing valuable information to arrive at a well-reasoned conclusion, ensuring that your own analysis remains primary. Conclude your response with \"the answer is \" followed by a single letter, without any additional punctuation.",
    "4": "You are an analytical problem solver dedicated to addressing complex reasoning tasks. As you consider responses from predecessor agents, evaluate each answer critically for relevance, clarity, and logical consistency. Differentiate between insights that enhance your understanding and those that may lead to confusion due to noise. Synthesize your conclusions by articulating a well-reasoned final response that reflects your independent reasoning while considering the collective insights provided. Present your answer as \"the answer is \" followed by a single letter, without any additional punctuation.",
    "5": "You are an analytical and independent thinker tasked with solving complex multiple-choice questions. While reviewing responses from predecessor agents, meticulously assess the relevance and accuracy of their insights, being vigilant for possible noise that may lead you astray. Use these insights judiciously to enhance your understanding, but prioritize your own reasoned analysis. Ensure that your final decision is based on a well-grounded evaluation of both your reasoning and pertinent insights, clearly articulating your choice. Conclude your response with \"the answer is \" followed by the correct letter, without any additional punctuation.",
    "6": "You are a discerning thinker dedicated to solving complex multiple-choice questions through rigorous analysis. Evaluate the responses from predecessor agents critically, focusing on extracting coherent insights while remaining vigilant for irrelevant or misleading information that may stem from noise. Prioritize your own logical reasoning and comprehensive understanding of the task when interpreting these insights. When discrepancies arise, ensure that your final decision is grounded in a robust analysis rather than solely reflecting the conclusions of others. Conclude your response with \"the answer is \" followed by your chosen letter, omitting any additional punctuation.",
    "7": "You are a discerning analytical thinker tasked with solving complex reasoning-based multiple-choice questions. As you review the responses from predecessor agents, extract any coherent and relevant insights but remain vigilant against misleading information. Critically compare these insights with your own logical reasoning, ensuring that your final answer is grounded in your independent analysis. Explicitly indicate where predecessor responses have influenced your thought process and where they diverged from your reasoning. Conclude your response with \"the answer is \" followed by a single letter.",
    "8": "You are an expert tasked with accurately resolving challenging reasoning-focused multiple-choice questions. Critically analyze the responses from predecessor agents, assessing their relevance and reliability in the context of the specific question. Clearly distinguish between insights that reinforce your understanding and those that may be misleading due to noise. Ensure your final answer is rooted in coherent reasoning, supported by logical analysis, rather than solely dependent on others\u2019 answers. Conclude your response with \"the answer is \" followed by a single letter representing your final choice without additional punctuation.",
    "9": "You are a highly capable agent specializing in complex reasoning-based questions. While analyzing your own logic and reasoning, carefully evaluate responses from predecessor agents to identify insights, but remain vigilant against potential noise or irrelevant information. Scrutinize their answers for any inconsistencies and determine how they may or may not contribute to refining your own conclusion. Use this evaluation selectively to enhance your answer, ensuring your final choice is evidence-backed and coherent. Conclude your response with the phrase \"the answer is\" followed by a single letter representing your final choice, without punctuation at the end.",
    "10": "As a refined reasoning agent, your primary objective is to analyze and solve complex multiple-choice questions through independent logical reasoning. While reviewing responses from predecessor agents, prioritize identifying consistent, relevant insights that enhance your understanding, but be cautious of misleading or irrelevant information. Critically evaluate the coherence and contribution of these insights and discern how they align with your own reasoning process. Synthesize the information judiciously, ensuring your final choice reflects sound judgment and independent thought. Conclude your response with \"the answer is \" followed by the letter of your selected option, without any additional punctuation or content."
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
        if i != j:
            adj_matrix[i, j] = 1


llm_configs = [qwen32b_config] * adj_matrix.shape[0] 

num_agents = len(llm_configs)
print('Agents:', num_agents)
print('Rounds:', num_rounds)
print('Edges:', np.sum(adj_matrix))

# multi agent时temperature设为1
temperature = 1.0
for c in llm_configs:
    c['temperature'] = temperature

################################# load data

file_path = '../data/MMLU_pro/mmlu_pro_test_128.csv'
data = pd.read_csv(file_path, encoding='utf8')
print(data.shape)

questions = data['question'].tolist()
choices = data['options'].tolist()
answers = data['answer'].tolist()

with open('../data/MMLU_pro/task2response.json', 'r') as f:
    single_task2messages = json.load(f)

def generate_random_answer():
    k = random.choice(list(single_task2messages.keys()))
    response = single_task2messages[k]
    return response

################################# prompt
prompt_template = """
Can you answer the following question as accurately as possible? {question_exam}: choices are in the [] below:{choice}. Please name those choices as A~Z in the order of the choices.Always remember that you can only give ana answer that exists.For example, if you're given ['1','2','3'],then please use A to represent '1',B to represent '2' and C to represent '3' .In this example,don't give answers besides A,B and C. The number of choices can vary from 4 to more.Explain your answer, and please strictly obey the rule that the last sentence of your response should be "the answer is " plus a signle letter.Do not put a dot at the end.
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

async def run_mmlu(error_rate):
    expname = f'{expname_prefix}_{error_rate}'
    print('Error rate:', error_rate)

    filepath = f"./output/MMLU/{expname}/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        print('make new dir',filepath)

    agents = []
    for i, config in enumerate(llm_configs):
        agent = ConversableAgent(
                    name=f'agent_{i}',
                    system_message=system_prompts[str(i+1)],
                    llm_config={"config_list": [config], 'temperature': temperature},
                    code_execution_config=False,
                    function_map=None,
                    max_consecutive_auto_reply=1000000,
                    human_input_mode="NEVER",
                )
        agents.append(agent)

    random.seed(0)
    task2messages = {}
    accumulated_usage = np.zeros((len(questions), num_rounds, num_agents, 2))

    task_id = 0
    for k,v in tqdm(list(enumerate(questions)), desc=f'Error_rate_{error_rate}'):
        # print(f"====={k}_{error_rate}=====")
        if k in task2messages:
            print('Existed!')
            continue

        question_exam = questions[k]
        choice = choices[k]
        question_prompt = prompt_template.format(question_exam=question_exam, choice=choice)

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
    accumulated_usage = accumulated_usage.reshape(len(questions)*num_rounds, num_agents, 2)
    token_usage = np.zeros(accumulated_usage.shape)
    token_usage[0] = accumulated_usage[0]
    for i in range(1, len(questions)*num_rounds):
        token_usage[i] = accumulated_usage[i] - accumulated_usage[i-1]
        
    accumulated_usage = accumulated_usage.reshape(len(questions), num_rounds, num_agents, 2)
    token_usage = token_usage.reshape(len(questions), num_rounds, num_agents, 2)

    # 保存token消耗和agent的原始回答
    np.save(os.path.join(filepath, 'token_usage.npy'), token_usage)
    with open(os.path.join(filepath, 'task2messages.json'), 'w') as f:
        json.dump(task2messages, f, indent=4)

    print("time:", time.time()-start_time)
    print('Saved to', filepath)

async def main():
    tasks = [run_mmlu(error_rate) for error_rate in [0.0, 0.2, 0.4, 0.6, 0.8]]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())