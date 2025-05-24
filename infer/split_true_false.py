import json
import os

file = "Qwen-2.5-32B-SimpleRL-Zoo_qwen-base-template_4096_infer_checked_correctedFalse"
with open('../save/inference/{}.json'.format(file), 'r', encoding='utf-8') as f:
    data = json.load(f)

save_true, save_false = [], []

for entry in data:
    if entry["is_correct"] == True:
        save_true.append(entry)
    else:
        continue
print('true saved: ',len(save_true))
with open('../save/inference/{}_all-true.json'.format(file), 'w', encoding='utf-8') as f:
    json.dump(save_true, f, ensure_ascii=False, indent=4)


for entry in data:
    if entry["is_correct"] == False:
        save_false.append(entry)
    else:
        continue
print('false saved: ',len(save_false))
with open('../save/inference/{}_all-false.json'.format(file), 'w', encoding='utf-8') as f:
    json.dump(save_false, f, ensure_ascii=False, indent=4)
