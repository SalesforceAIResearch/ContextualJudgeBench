#!/bin/bash

cd <CONTEXTUALJUDGEBENCH_DIR>
export OPENAI_API_KEY=<YOUR_OPENAI_TOKEN_HERE>

#Just used to load judgment parsing code
judge_model=FULL_JUDGE_PATH_OR_HF_NAME
judge_model_short=NAME_FOR_OUTPUT_PATH_NAMING
prompt_strategy=vanilla

# eval_dir = directory where .json(l) output files are stored for judge
eval_dir=/path/to/your/output/dir/${judge_model_short}/${prompt_strategy}
python main_eval_critiques.py \
    --eval_dir ${eval_dir} \
    --judge_model ${judge_model} \


