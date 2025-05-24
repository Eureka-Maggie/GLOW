export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_HOME="/path/to/huggingface"
huggingface-cli login --token token
export WANDB_MODE=online
export WANDB_PROJECT="SFT"
wandb login --relogin token

llamafactory-cli train examples/weak2strong/qwen2.5-14b/14brl-sft_base.yaml