export CUDA_VISIBLE_DEVICES='0,1,2,3' # MUST have this
export HF_HOME="/path/to/huggingface"
huggingface-cli login --token token

DATAS=("math" "amc") 
KS=({1..128}) 
declare -A MODELS=(
    ["name"]="path/to/checkpoints"
)

TEMPS=(0.6)
for model_key in "${!MODELS[@]}"; do
    for data_name in "${DATAS[@]}"; do
        for tmp in "${TEMPS[@]}"; do

            IFS=',' read -r model_name_or_path <<< "${MODELS[$model_key]}"

            echo "Running Acc evaluation for data: $data_name, model: $model_key"
            echo "Model Path: $model_name_or_path"

            if [[ "$model_key" != *"hf"* ]] && [ -z "$(find "${model_name_or_path}" -name '*.safetensors' -print -quit)" ]; then
                echo "Merging Model.."
                python model_merger.py --local_dir "$(echo "$model_name_or_path" | sed 's|/[^/]*$||')"
            fi
            
            model_name_or_path="$model_name_or_path"
            echo "Eval $model_name_or_path"

            PATTERN="qwen-instruct-template|qwen-base-template|simple-template|cft-template"
            PROMPT=$(echo "$model_key" | grep -oE "$PATTERN" | head -n 1)

            python eval_passk.py \
                --model_name_or_path "$model_name_or_path" \
                --data_name "$data_name" \
                --save_name "$model_key" \
                --prompt_type $PROMPT \
                --temperature $tmp \
                --start_idx 0 \
                --end_idx -1 \
                --n_sampling 1 \
                --k 1 \
                --split "test" \
                --max_tokens 32768 \
                --seed 0 \
                --top_p 0.95 \
                --surround_with_messages \
                --vllm_memory 0.9
        done
    done
done

