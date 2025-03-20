import json, os, time, argparse, importlib, glob, sys
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset, load_dataset

from utils.prompt_utils import (
        load_prompts_and_parsing,
        OPENAI_MODEL_LIST,
        API_MODEL_LIST,
    )

from utils.critique_eval_utils import CRITIQUE_EVAL_PROMPT, criteria_dict_critique_eval
from utils.utils import chat_completion_openai


HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def main(args):

    # Check if judge is queried via API 
    is_api_models = isinstance(args.evaluator, list) or args.evaluator in API_MODEL_LIST
    assert is_api_models # Assumption is you use GPT-4o

    # Load model evaluations
    all_eval_outputs = glob.glob(os.path.join(args.eval_dir, "eval_detailed*.jsonl"))
    if all_eval_outputs == []:
        print('No evaluation outputs found! Did you pass the right directory?')
        return

    # Load judge parsing -- judge_prompt unused
    judge_prompt, get_judgement = load_prompts_and_parsing(args.judge_model, "vanilla") #Only run eval for vanilla
    if judge_prompt == '':
        raise Exception('No judge prompt loaded!')

    for eval_split_outputs in all_eval_outputs:
        if 'metaeval' in eval_split_outputs:
            continue

        if os.path.exists(eval_split_outputs.replace('.jsonl', '.metaeval.jsonl')):
            print(f'Already evaluated, skipping {eval_split_outputs}')
            continue

        print(f"Evaluating: [{eval_split_outputs}]")
        eval_dataset = load_dataset("json", data_files=eval_split_outputs, split='train')
        print(eval_dataset)

        if args.debug:
            eval_dataset = eval_dataset.select(range(10))

        
        # Set the evaluation criteria based on prompt strategy and split
        evaluation_criteria = ''

        candidate_splits = ['conciseness', 'completeness', 'refusal', 'factuality']
        assert any(cs in eval_split_outputs for cs in candidate_splits), f'Split not in {candidate_splits}'
        
        for cs in candidate_splits:
            if cs in eval_split_outputs:
                evaluation_criteria = criteria_dict_metaeval[cs]
                break
        

        # evaluation_criteria = criteria_dict_metaeval['faithfulness']
        assert evaluation_criteria != '', 'Evaluation criteria empty! Invalid split and prompt strategy combination'

        print(f"Using criteria: {evaluation_criteria}")
        # Inference!

        # Implementation adopted from RewardBench: https://github.com/allenai/reward-bench/blob/main/scripts/run_generative.py
        if args.evaluator in OPENAI_MODEL_LIST:
            chat_completion = chat_completion_openai
        
        def update_progress_bar(done, total):
            # Simple text-based progress bar
            progress = int(50 * done / total)  # Calculate progress (50 chars width)
            sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
            sys.stdout.flush()

        def get_api_output(example, idx):
            # Model got it incorrect
            if example['votes'][idx-1] != 1:
                return None
            
            judge_output = example[f'swap_inference{idx}']
            flip = (idx == 2)
            label = 'Response A' if not flip else 'Response B'
            if 'Self-taught-evaluator' in args.judge_model:
                # No good programatic way of just critique output from self-taught
                # Just use entire output as critique
                critique = judge_output 
            else:
                _, critique = get_judgement(judge_output, return_critique=True)
                

            content = CRITIQUE_EVAL_PROMPT.format(
                judgment=label,
                critique=critique,
                criteria=evaluation_criteria
            )

            messages = [
                {"role": "user", "content": content}
            ]

            # query api
            output = chat_completion(args.evaluator, messages, args.temperature, 1024)

            # return output
            return output

        pair_inferences = []

        # Run each eval twice for each critique
        for idx in [1, 2]:
            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                inferences_idx = [None] * len(eval_dataset) # Preallocate output list
                done_tasks = 0

                future_to_index = {executor.submit(get_api_output, x, idx): i for i, x in enumerate(eval_dataset)}

                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    inferences_idx[index] = future.result()
                    done_tasks += 1
                    update_progress_bar(done_tasks, len(eval_dataset))

                pair_inferences.append(inferences_idx)
            print()

        
        votes_evald = []
        reasons_evald = []
        for out1, out2 in zip(pair_inferences[0], pair_inferences[1]):
            
            decision1 =  None

            if out1 is not None:
                o1 = out1.split('<decision>')[-1]
                if 'yes' in o1.lower() and 'no' not in o1.lower():
                    decision1 = 1
                elif 'no' in o1.lower() and 'yes' not in o1.lower():
                    decision1 = 0
                else:
                    decision1 = 1 # Benefit of the doubt

            decision2 = None
            if out2 is not None:
                o2 = out2.split('<decision>')[-1]
                if 'yes' in o2.lower() and 'no' not in o2.lower():
                    decision2 = 1
                elif 'no' in o2.lower() and 'yes' not in o2.lower():
                    decision2 = 0
                else:
                    decision2 = 1 # Benefit of the doubt

            votes_evald.append([decision1, decision2])
            reasons_evald.append([out1, out2])
        
        eval_dataset = eval_dataset.add_column("votes_evald", votes_evald)
        eval_dataset = eval_dataset.add_column("reasons_evald", reasons_evald)
        
        eval_dataset = eval_dataset.select_columns(['votes', 'votes_evald', 'reasons_evald'])
        output_path = eval_split_outputs.replace('.jsonl', '.critique_eval.jsonl')
        eval_dataset.to_json(output_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Evaluation parameters
    parser.add_argument("--eval_dir", type=str, default="./")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--num_threads", type=int, default=10, help="number of threads to use for parallel processing of examples for openai api calls")

    # judge info
    parser.add_argument("--judge_model", type=str, help="model checkpoint or name")
    parser.add_argument("--evaluator", type=str, default="gpt-4o")

    # decoding strategy
    parser.add_argument("--temperature", default=0.0, type=float)

    args = parser.parse_args()

    main(args)