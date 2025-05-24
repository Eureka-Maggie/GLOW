import json
import sys

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re
import importlib.util
import os
import argparse
import vllm.envs as envs
import random
import time
from datetime import datetime
from tqdm import tqdm

from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.math_normalization import *
from utils.grader import *
import pickle
from math import comb


# 模型名称
file = "Llama-3.1-8B-SimpleRL-Zoo_simple-template_4096_infer"
do_correction=False
model_name = "Qwen/Qwen2.5-0.5B"

def get_conversation_prompt_by_messages(tokenizer, messages):
    if (messages[0]['role'] == 'system') and (messages[0]['content'] == 'none'):
        text = "Question:\n{question}\nAnswer:\nLet's think step by step. ".format(question=messages[-1]['content'])
    else:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    return text

def make_conversation_instruct(example, q_key):
    return [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": example[q_key]},
        ]

def make_conversation_base(example, q_key):
    return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example[q_key]+'\n'+"Please reason step by step, and put your final answer within \\boxed{}."},
        ]

def make_conversation_simple(example, q_key):
    return [
            {"role": "system", "content": "none"},
            {"role": "user", "content": example[q_key]},
        ]

if do_correction:
    # # 初始化 vLLM
    # llm =None
    llm = LLM(
        model=model_name,
        tensor_parallel_size=2,  # 设置为你的 GPU 数量
        max_model_len=8192,  # 设置 max_model_len 为 8192 以支持 8K 上下文
        gpu_memory_utilization=0.8,
        trust_remote_code=True
    )

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(temperature=1, top_p=1, max_tokens=3000, n=16)


# 加载原始 JSON 文件
with open('../save/inference/{}-1-0.0.json'.format(file), 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('math_level3to5_data_processed_with_qwen_prompt.json', 'r', encoding='utf-8') as f:
    origin = json.load(f)

if ("question" in data[0]) and ("response" in data[0]):
    q_key = "question"
    a_key = "response"
elif ("instruction" in data[0]) and ("output" in data[0]):
    q_key = "instruction"
    a_key = "output"
else:
    raise

save=[]
for entry, origin_entry in tqdm(zip(data, origin), desc="Processing entries"):
    
    if type(entry[a_key]) is str:
        ori_response = entry[a_key] 
    elif type(entry[a_key]) is list:
        ori_response = entry[a_key][0]
    else:
        raise
    ret_response = ori_response
    gt_answer = origin_entry["gt_answer"]
    
    if "is_correct" in entry:
        is_correct = entry["is_correct"]
    else:
        model_answer  = extract_answer(ori_response, "math")  
        print('model_answer', model_answer)
        is_correct = check_is_correct(model_answer, gt_answer)
        print('is_correct', is_correct)
    
    if (not is_correct) and (do_correction):
        _p = make_conversation_instruct(entry, q_key)
        prompt = get_conversation_prompt_by_messages(tokenizer, _p)
        outputs = llm.generate(prompt, sampling_params)
        
        responses = [output.outputs[0].text for output in outputs]
        correct_response = None  
        for response in responses:
            generated_answer = extract_answer(response, "math")  
            if check_is_correct(generated_answer, origin_entry["gt_answer"]): 
                correct_response = response  
                break  

        if correct_response is not None:
            is_correct = True
            ret_response = correct_response
        else:
            is_correct = False
            ret_response = ori_response

    ret = { "instruction": entry[q_key],
            "output": ret_response,
            "is_correct": is_correct,
            "gt_answer": gt_answer
            }
    save.append(ret)

    # 保存更新后的 JSON 文件（保持原始顺序）
    with open('../save/inference/{}_checked_corrected{}.json'.format(file, do_correction), 'w', encoding='utf-8') as f:
        json.dump(save, f, ensure_ascii=False, indent=4)

print("Re-inference and correctness check completed. Updated JSON file saved.")