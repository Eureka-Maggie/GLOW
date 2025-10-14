import os
import json
import random
from datasets import load_dataset, Dataset, concatenate_datasets
from .utils import load_jsonl, lower_keys

def load_data(data_name, split, data_dir='./data'):
    data_file = f"{data_dir}/{data_name}/{split}.jsonl"
    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))
    else:
        if data_name == "math":
            dataset = load_dataset("competition_math", split=split, name="main", cache_dir=f"{data_dir}/temp")
        elif data_name == "theorem-qa":
            dataset = load_dataset("wenhu/TheoremQA", split=split)
        elif data_name == "gsm8k":
            dataset = load_dataset(data_name, split=split)
        elif data_name == "gsm-hard":
            dataset = load_dataset("reasoning-machines/gsm-hard", split="train")
        elif data_name == "svamp":
            # evaluate on training set + test set 
            dataset = load_dataset("ChilleD/SVAMP", split="train")
            dataset = concatenate_datasets([dataset, load_dataset("ChilleD/SVAMP", split="test")])
        elif data_name == "asdiv":
            dataset = load_dataset("EleutherAI/asdiv", split="validation")
            dataset = dataset.filter(lambda x: ";" not in x['answer']) # remove multi-answer examples
        elif data_name == "mawps":
            examples = []
            # four sub-tasks
            for data_name in ["singleeq", "singleop", "addsub", "multiarith"]:
                sub_examples = list(load_jsonl(f"{data_dir}/mawps/{data_name}.jsonl"))
                for example in sub_examples:
                    example['type'] = data_name
                examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        elif data_name == "finqa":
            dataset = load_dataset("dreamerdeo/finqa", split=split, name="main")
            dataset = dataset.select(random.sample(range(len(dataset)), 1000))
        elif data_name == "tabmwp":
            examples = []
            with open(f"{data_dir}/tabmwp/tabmwp_{split}.json", "r") as f:
                data_dict = json.load(f)
                examples.extend(data_dict.values())
            dataset = Dataset.from_list(examples)
            dataset = dataset.select(random.sample(range(len(dataset)), 1000))
        elif data_name == "bbh":
            examples = []
            # for data_name in ["test"]:
            for data_name in ["boolean_expressions","causal_judgement","date_understanding","disambiguation_qa",\
                             "formal_fallacies","geometric_shapes","hyperbaton","logical_deduction_five_objects",\
                             "logical_deduction_seven_objects","logical_deduction_three_objects","movie_recommendation","multistep_arithmetic_two","navigate",\
                             "object_counting","penguins_in_a_table","reasoning_about_colored_objects","ruin_names","snarks","sports_understanding",\
                             "temporal_sequences","tracking_shuffled_objects_five_objects","tracking_shuffled_objects_seven_objects","tracking_shuffled_objects_three_objects","web_of_lies","word_sorting"]:
                with open(f"{data_dir}/bbh/{data_name}.json", "r") as f:
                    sub_examples = json.load(f)
                    for example in sub_examples:
                        example['type'] = data_name
                    examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        elif data_name == "bbh_dyck":
            examples = []
            with open(f"{data_dir}/bbh_dyck/dyck_languages.json", "r") as f:
                sub_examples = json.load(f)
                for example in sub_examples:
                    example['type'] = data_name
                examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        elif data_name == "IFeval":
            with open(f"{data_dir}/ifeval/ifeval_format.json", "r") as f:
                examples = list(load_jsonl(f))
            dataset = Dataset.from_list(examples)
        else:
            raise NotImplementedError(data_name)

        examples = list(dataset)
        examples = [lower_keys(example) for example in examples]
        dataset = Dataset.from_list(examples)
        os.makedirs(f"{data_dir}/{data_name}", exist_ok=True)
        dataset.to_json(data_file)

    # add 'idx' in the first column
    if 'idx' not in examples[0]:
        examples = [{'idx': i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x['idx'])
    return examples