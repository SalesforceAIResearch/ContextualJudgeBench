#!/bin/bash

cd <CONTEXTUALJUDGEBENCH_DIR>
export HF_TOKEN=<YOUR_HF_TOKEN_HERE>
export OPENAI_API_KEY=<YOUR_OPENAI_TOKEN_HERE>

judge_models=(
    gpt-4o-2024-08-06
    gpt-4o-mini-2024-07-18
    o1
    o3-mini
)

judge_models_short=(
    gpt-4o-2024-08-06
    gpt-4o-mini-2024-07-18
    o1
    o3-mini
)

splits=(
    "all"
    # completeness_qa
    # completeness_summ
    # conciseness_qa
    # conciseness_summ
    # faithfulness_qa
    # faithfulness_summ
    # refusal_answerable
    # refusal_unanswerable
)

for i in "${!judge_models[@]}"; do
    judge_model=${judge_models[i]}
    judge_model_short=${judge_models_short[i]}
    output_path=/path/to/your/output/dir/${judge_model_short}/

    for si in "${!splits[@]}"; do
        split=${splits[si]}
        
        # Prompt used in main table
        python main_eval.py \
        --output_path ${output_path} \
        --judge_model ${judge_model} \
        --prompt_strategy vanilla \
        --splits ${split} \

        # # Prompt used in Sec. 4.3
        # python main_eval.py \
        # --output_path ${output_path} \
        # --judge_model ${judge_model} \
        # --prompt_strategy criteria_specific \
        # --splits ${split} \

        # # Prompt used in Appendix C
        # python main_eval.py \
        # --output_path ${output_path} \
        # --judge_model ${judge_model} \
        # --prompt_strategy conditional \
        # --splits ${split} \

    done

done

