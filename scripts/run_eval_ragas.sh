cd <CONTEXTUALJUDGEBENCH_DIR>
export HF_TOKEN=<YOUR_HF_TOKEN_HERE>
export OPENAI_API_KEY=<YOUR_OPENAI_TOKEN_HERE>

# RAGAS eval
split="all"
judge_model=ragas
judge_model_short=ragas
output_path=/path/to/your/output/dir/${judge_model_short}/
python main_eval.py \
    --output_path ${output_path} \
    --judge_model ${judge_model} \
    --splits ${split} \


