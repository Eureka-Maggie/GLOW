import os
import json
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import vllm.envs as envs

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with vLLM")
    parser.add_argument("--model_name", type=str, default="hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo",
                       help="Name of the model to use for inference")
    parser.add_argument("--input_file", type=str, default="math_level3to5_data_processed_with_qwen_prompt.json",
                       help="Path to input JSON file")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Path to output JSON file")
    parser.add_argument("--template_type", type=str, default='qwen-instruct-template',
                       help="Template")
    parser.add_argument("--max_model_len", type=int, default=4096,
                       help="Maximum model length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for inference")
    parser.add_argument("--n", type=int, default=8,
                       help="n")
    parser.add_argument("--temperature", type=float, default=1,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p sampling value")
    parser.add_argument("--max_tokens", type=int, default=4096,
                       help="Maximum tokens to generate")
    return parser.parse_args()

def get_conversation_prompt_by_messages(tokenizer, messages):
    if tokenizer.chat_template is None:
        tokenizer.chat_template = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B").chat_template

    if (messages[0]['role'] == 'system') and (messages[0]['content'] == 'none'):
        text = "Question:\n{question}\nAnswer:\nLet's think step by step. ".format(question=messages[-1]['content'])
    else:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    return text

def load_and_preprocess_data(input_file):
    dataset = DatasetDict()
    dataset["train"] = load_dataset('json', data_files=input_file, split='train')
    train_columns_to_remove = ['ground_truth_answer', 'target', 'input', 'subject', 'level']
    dataset["train"] = dataset["train"].rename_columns({
        "question": "problem",
        "answer": "solution"
    })
    
    dataset["train"] = dataset["train"].remove_columns(train_columns_to_remove)
    return dataset

def make_conversation_instruct(example):
    return {
        "prompt": [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": example["problem"]},
        ],
    }

def make_conversation_base(example):
    return {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example["problem"]+'\n'+"Please reason step by step, and put your final answer within \\boxed{}."},
        ],
    }

def make_conversation_simple(example):
    return {
        "prompt": [
            {"role": "system", "content": "none"},
            {"role": "user", "content": example["problem"]},
        ],
    }

def main():
    args = parse_args()
    
    # Set output file path if not provided
    if args.output_file is None:
        os.makedirs('../save/inference', exist_ok=True)
        args.output_file = f"../save/inference/{args.model_name.split('/')[-1]}_{args.template_type}_{args.max_model_len}_infer-{args.n}-{args.temperature}.json"
    
    # Load existing results if output file exists
    output_data = []
    processed_indices = set()
    processed_questions=None
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            output_data = json.load(f)
            # Get indices of already processed questions
            processed_questions = {item["question"] for item in output_data}
            processed_indices = set()  # We'll populate this after loading the dataset

    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')

    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=len(available_gpus),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_and_preprocess_data(args.input_file)

    if args.template_type == 'qwen-instruct-template':
        dataset = dataset.map(make_conversation_instruct)
    elif args.template_type == 'qwen-base-template':
        dataset = dataset.map(make_conversation_base)
    elif args.template_type == 'simple-template':
        dataset = dataset.map(make_conversation_simple)
    else:
        raise
    print(dataset)

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n,
        skip_special_tokens=True
    )

    # Get indices of already processed questions
    if processed_questions:
        all_questions = [item["prompt"][-1]["content"] for item in dataset["train"]]
        processed_indices = {i for i, q in enumerate(all_questions) if q in processed_questions}

    # Process data in batches
    for i in tqdm(range(0, len(dataset["train"]), args.batch_size)):
        # Skip already processed items in this batch
        batch_indices = range(i, min(i + args.batch_size, len(dataset["train"])))
        if processed_indices and all(idx in processed_indices for idx in batch_indices):
            print('continue')
            continue
            
        batch = dataset["train"][i:i + args.batch_size]
        raw_prompts = batch["prompt"]
        problems = batch["problem"]
        gt_answers = batch["gt_answer"]
        
        # Filter out already processed items
        if processed_indices:
            filtered_prompts = []
            filtered_gt_answers = []
            filtered_problems = []
            filtered_indices = []
            
            for j in range(len(raw_prompts)):
                if i + j not in processed_indices:
                    filtered_prompts.append(raw_prompts[j])
                    filtered_gt_answers.append(gt_answers[j])
                    filtered_problems.append(problem[j])
                    filtered_indices.append(i + j)
            
            if not filtered_prompts:  # All items in this batch are already processed
                continue
                
            raw_prompts = filtered_prompts
            gt_answers = filtered_gt_answers
            problems = filtered_problems

        # Perform batch inference
        batch_prompts = [get_conversation_prompt_by_messages(tokenizer, prompt) for prompt in raw_prompts]
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        # Decode generated text
        batch_responses = []
        batch_logps = []
        for q in outputs:
            oa, op = [], []
            for o in q.outputs:
                oa.append(o.text)
                op.append(o.cumulative_logprob)
            batch_responses.append(oa)
            batch_logps.append(op)

        # Save results for current batch
        for idx, (raw, gt_answer, problem, prompt, responses, logps) in enumerate(zip(
            raw_prompts, gt_answers, problems, batch_prompts, batch_responses, batch_logps
        )):
            result = {
                "prompt": prompt,
                "question": problem,
                "response": responses,
                "gt_answer": gt_answer
            }
            
            if processed_indices:
                # Insert at the correct position if we're continuing from a previous run
                actual_index = filtered_indices[idx]
                if actual_index < len(output_data):
                    output_data[actual_index] = result
                else:
                    output_data.append(result)
            else:
                output_data.append(result)

        # Save results to JSON file after each batch
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Inference completed. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()