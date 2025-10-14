export CUDA_VISIBLE_DEVICES='0,1,2,3' # MUST have this
#export HF_HOME="/path/to/huggingface"
#huggingface-cli login --token token

DATAS=("math" "minerva" "amc" "olympiadbench" "gpqa" "bbh" "mmlu" "mmlu_pro" "bbh_dyck" ) 
#KS=({1..128}) 
declare -A MODELS=(
    ["qwen2p5_3_MMLU_all_20e"]="model_path",
    ["qwen2p5_3_MMLU_correct_20e"]="model_path",
)

TEMPS=(0.6)
for model_key in "${!MODELS[@]}"; do
    for data_name in "${DATAS[@]}"; do
        for tmp in "${TEMPS[@]}"; do

            IFS=',' read -r model_name_or_path <<< "${MODELS[$model_key]}"

            echo "Running Acc evaluation for data: $data_name, model: $model_key"
            echo "Model Path: $model_name_or_path"
            
            model_name_or_path="$model_name_or_path"
            echo "Eval $model_name_or_path"

            if [[ "$model_key" == *"ins"* ]]; then
                PROMPT="qwen-instruct-template"
            elif [[ "$model_key" == *"0.5B"* || "$model_key" == *"1.5B"* ]]; then
                PROMPT="simple-template"
            else
                PROMPT="qwen-base-template"
            fi
            echo "Selected prompt template: $PROMPT"

            python eval_passk.py \
                --model_name_or_path "$model_name_or_path" \
                --data_name "$data_name" \
                --save_name "$model_key" \
                --prompt_type $PROMPT \
                --temperature 0.6 \
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

