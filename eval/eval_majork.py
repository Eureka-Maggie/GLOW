from utils.grader import math_equal
from utils.parser import strip_string
import timeout_decorator
from collections import defaultdict, Counter
from utils.utils import set_seed,load_jsonl
import argparse
import os
import glob
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="", help="model_name")
    parser.add_argument('--data_name', type=str, default="", help="data_name")
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--seed", default=0, type=int)
    # parser.add_argument("--use_qwen_check", action="store_true")
    args = parser.parse_args()
    return args


@timeout_decorator.timeout(5)
def math_equal_timeout(pred, gt):
    try:
        return math_equal(pred, gt)
    except Exception as e:
        print("Timeout error:", e)
        return False


def group_pred(preds, strip=True, use_symbol=False):
    print(len(preds))
    orginal_preds = preds
    if not use_symbol:
        if strip:
            preds = [strip_string(pred) for pred in preds]
        cnt = Counter(preds)
        majority = cnt.most_common(1)[0][0]
        groups = defaultdict(list)
        for idx, pred in enumerate(preds):
            groups[pred].append(idx)
        return groups, orginal_preds[groups[majority][0]]

    groups = defaultdict(list)
    for idx, pred in enumerate(preds):
        found_group = False
        if strip:
            pred = strip_string(pred)
        for group_pred in groups:
            try:
                if math_equal_timeout(pred, group_pred):
                    groups[group_pred].append(idx)
                    found_group = True
                    break
            except:
                continue
        if not found_group:
            groups[pred].append(idx)
    # get the key of the longest group
    majority = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)[0][0]
    majority = orginal_preds[groups[majority][0]]
    return groups, majority


def eval_rm_k_metrics(data_path, k=8):
    print(f"evaluating rm@{k}")
    data_list = load_jsonl(data_path)

    count, right_count = 0, 0
    for sample in data_list:
        assert len(sample['pred_score']) >= k, sample['data_source']
        pred_score = sample['pred_score'][:k]
        pred = sample['score'][:k]
        assert len(pred_score) == len(pred), f"{len(pred_score)}, {len(pred)}"

        rm_score = pred_score
        rm_score = [inner_score for score in rm_score for inner_score in score]
        assert len(rm_score) == len(pred), f"{len(rm_score)}, {len(pred)}"

        max_index = rm_score.index(max(rm_score))
        max_pred = pred[max_index]
        right_count += max_pred
        count += 1

    print(count)
    task_acc = right_count / count * 100
    print(f"acc: {task_acc:.1f}")
    return task_acc


def eval_maj_k_metrics(jsonl_path_list, result_path, k=8):
    print(f"evaluating maj@{k}")
    #print('jsonl_path',jsonl_path_list)
    for i in jsonl_path_list:
        if ('k128' in i):
            jsonl_path=i
    print('jsonl_path',jsonl_path)
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data_list = [json.loads(line.strip()) for line in f]
    print('load done')
    # data_list = load_jsonl(jsonl_path)
    count, right_count = 0, 0
    for sample in data_list:
        assert len(sample['answers_correctness']) >= k, sample
        groups, majority_pred = group_pred(sample['generated_answers'][:k], strip=False, use_symbol=False)
        idx = groups[majority_pred][0]
        right_count += int(sample['answers_correctness'][idx])
        count += 1


    task_acc = right_count / count 
    with open(result_path, "a") as f:
        f.write(f"---------------------------\n")
        f.write(f"Maj@{k}: {right_count}/{count} = {task_acc:.4f}\n")
    # print(f"acc: {task_acc:.1f}")
    return task_acc


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    save_path = "./outputs/{}/{}".format(args.model_name, args.data_name)
    print('save_path',save_path)
    pattern = os.path.join(save_path, '*.jsonl')
    jsonl_path = glob.glob(pattern)
    result_path = "{}/result.txt".format(save_path)
    candidate = args.k
    # all_result = {}
    majk = eval_maj_k_metrics(jsonl_path, result_path, k=candidate)
    # all_result[f'rm@{candidate}'] = eval_rm_k_metrics(data_path, k=candidate)
    # print(all_result)
