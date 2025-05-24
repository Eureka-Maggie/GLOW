export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_HOME="/path/to/huggingface"
huggingface-cli login --token token

python generate.py --model_name "hkust-nlp/Mistral-Small-24B-SimpleRL-Zoo" --template_type "qwen-base-template" --n 1 --temperature 0


